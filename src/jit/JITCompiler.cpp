#include "JITCompiler.h"
#include "MatMulKernel.h"
#include <fstream>

#include <cuda.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <iostream>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/IR/LegacyPassManager.h>   // <- REQUIRED for llvm::legacy::PassManager
#include <llvm/Support/CodeGen.h>        // <- for llvm::CGFT_AssemblyFile
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/IPO/Internalize.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/IPO.h>                 // (createNVVMReflectPass in some LLVMs)
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>


#define CUDA_DRV_CHECK(call)                                                     \
  do {                                                                           \
    CUresult _e = (call);                                                        \
    if (_e != CUDA_SUCCESS) {                                                    \
      const char* _errStr = nullptr;                                             \
      cuGetErrorString(_e, &_errStr);                                            \
      std::cerr << "CUDA Driver API error " << _e << " ("                        \
                << (_errStr ? _errStr : "unknown")                               \
                << ") at " << __FILE__ << ":" << __LINE__ << std::endl;          \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

using namespace llvm;
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>
namespace {

static std::string emitPTX(llvm::Module &M, llvm::TargetMachine &TM) {
    // ---- New PM for optimization (optional) ----
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PassBuilder PB(&TM);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    for (auto &F : M) {
        for (auto &BB : F) {
            for (auto &I : BB) {
            if (auto *CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
                auto Callee = CB->getCalledOperand();
                if (auto *FT = CB->getFunctionType()) {
                // Compare number/types of params
                for (unsigned i = 0; i < CB->arg_size(); ++i) {
                    if (i >= FT->getNumParams() ||
                        FT->getParamType(i) != CB->getArgOperand(i)->getType()) {
                    llvm::errs() << "\n*** BAD CALL *** in " << F.getName() << "\n";
                    CB->print(llvm::errs()); llvm::errs() << "\n";
                    if (auto *FDecl = llvm::dyn_cast<llvm::Function>(Callee))
                        FDecl->print(llvm::errs()), llvm::errs() << "\n";
                    std::exit(1);
                    }
                }
                }
            }
            }
        }
    }


    llvm::ModulePassManager MPM;
    MPM.addPass(llvm::InternalizePass([&](const llvm::GlobalValue &GV) {
        return GV.getName() == "matmul"; // keep kernel
    }));
    // O2 optimizations
    MPM.addPass(PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2));
    //auto MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
    MPM.run(M, MAM);

    if (!M.getNamedMetadata("nvvm.annotations")) {
        llvm::Metadata *mdVals[] = {
            llvm::ValueAsMetadata::get(M.getFunction("matmul")),
            llvm::MDString::get(M.getContext(), "kernel"),
            llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(Type::getInt32Ty(M.getContext()), 1))
        };
        auto *mdNode = llvm::MDNode::get(M.getContext(), mdVals);
        M.getOrInsertNamedMetadata("nvvm.annotations")->addOperand(mdNode);
    }

    // ---- Legacy PM for codegen to PTX ----
    llvm::SmallVector<char, 0> buffer;
    llvm::raw_svector_ostream os(buffer);   // <-- this IS a raw_pwrite_stream

    llvm::legacy::PassManager codegenPM;
    if (TM.addPassesToEmitFile(codegenPM, os, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
        llvm::report_fatal_error("TargetMachine can't emit PTX");
    }
    codegenPM.run(M);
    // os auto-flushes on destruction

    return std::string(buffer.begin(), buffer.end());
}


} // namespace

JITCompiler::JITCompiler() {
    // Initialize only X86 and NVPTX
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();

    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
}

void JITCompiler::initialize() {
    auto Context = std::make_unique<llvm::orc::ThreadSafeContext>(std::make_unique<llvm::LLVMContext>());
    auto JIT = llvm::orc::LLJITBuilder().create();
    if (!JIT) {
        std::cerr << "Failed to create JIT: " << toString(JIT.takeError()) << std::endl;
        exit(1);
    }
    jit = std::move(*JIT);
}

void JITCompiler::compileAndExecute(
    const std::string &kernelCode,
    float *left, float *right, float *result,
    int rows, int columns, int inners) {

    // 1. Create a ThreadSafeContext and extract reference
    auto tsCtx = llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());
    llvm::LLVMContext &ctx = *tsCtx.getContext();

    // 2. Create your kernel module
    auto matMulKernel = std::make_unique<MatMulKernel>();
    //llvm::orc::ThreadSafeModule TSM = matMulKernel->createModule(ctx);
    llvm::orc::ThreadSafeModule TSM = matMulKernel->createModule(tsCtx);

    std::string ptx;

    // 3. Generate PTX
    llvm::Error genErr = TSM.withModuleDo([&](llvm::Module &OrigM) -> llvm::Error {
        // --- Clone module (so new PM won't break ThreadSafeModule's context) ---
        auto Cloned = llvm::CloneModule(OrigM);
        llvm::Module &M = *Cloned;

        // --- Setup target ---
        std::string errStr;
        const llvm::Target *target = llvm::TargetRegistry::lookupTarget("nvptx64", errStr);
        if (!target) {
            std::cerr << "Failed to lookup NVPTX target: " << errStr << "\n";
            return llvm::make_error<llvm::StringError>("lookupTarget failed",
                                                       llvm::inconvertibleErrorCode());
        }

        llvm::TargetOptions opt;
        std::optional<llvm::Reloc::Model> relocModel;
        llvm::Triple TT("nvptx64-nvidia-cuda");
        std::unique_ptr<llvm::TargetMachine> TM(
            target->createTargetMachine(TT,
                                        "sm_89", "", opt, relocModel));
        if (!TM) {
            std::cerr << "Failed to create NVPTX TargetMachine\n";
            return llvm::make_error<llvm::StringError>("createTargetMachine failed",
                                                       llvm::inconvertibleErrorCode());
        }

        M.setDataLayout(TM->createDataLayout());

        // --- Link libdevice BEFORE optimization ---
        llvm::SMDiagnostic diag;
        auto libDevice = llvm::parseIRFile("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc", diag, M.getContext());
        if (!libDevice) {
            diag.print("jit", llvm::errs());
            return llvm::make_error<llvm::StringError>("parse libdevice failed",
                                                       llvm::inconvertibleErrorCode());
        }
        if (llvm::Linker::linkModules(M, std::move(libDevice))) {
            return llvm::make_error<llvm::StringError>("link with libdevice failed",
                                                       llvm::inconvertibleErrorCode());
        }

        // --- Verify before optimization ---
        if (llvm::verifyModule(M, &llvm::errs())) {
            llvm::errs() << "<<< Invalid module BEFORE optimization >>>\n";
            M.print(llvm::errs(), nullptr);
            std::exit(1);
        }

        // --- Optimize with new PM ---
        {
            llvm::LoopAnalysisManager LAM;
            llvm::FunctionAnalysisManager FAM;
            llvm::CGSCCAnalysisManager CGAM;
            llvm::ModuleAnalysisManager MAM;

            llvm::PassBuilder PB(TM.get());
            PB.registerModuleAnalyses(MAM);
            PB.registerCGSCCAnalyses(CGAM);
            PB.registerFunctionAnalyses(FAM);
            PB.registerLoopAnalyses(LAM);
            PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

            auto MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
            MPM.run(M, MAM);
        }

        // --- Verify after optimization ---
        if (llvm::verifyModule(M, &llvm::errs())) {
            llvm::errs() << "<<< Invalid module AFTER optimization >>>\n";
            M.print(llvm::errs(), nullptr);
            std::exit(1);
        }

        // --- Emit PTX ---
        ptx = emitPTX(M, *TM);

        // Write PTX for debugging
        std::ofstream ofs("kernel.ptx");
        ofs << ptx;
        ofs.close();

        return llvm::Error::success();
    });

    if (genErr) {
        std::cerr << "PTX generation error: " << llvm::toString(std::move(genErr)) << "\n";
        std::exit(1);
    }

    // -----------------------------------
    // CUDA Driver API: load & launch
    // -----------------------------------
    CUDA_DRV_CHECK(cuInit(0));
    CUdevice dev;
    CUDA_DRV_CHECK(cuDeviceGet(&dev, 0));
    CUcontext cuCtx;
    CUDA_DRV_CHECK(cuCtxCreate(&cuCtx, 0, dev));

    CUmodule cuMod;
    CUDA_DRV_CHECK(cuModuleLoadData(&cuMod, ptx.c_str()));

    CUfunction cuFunc;
    CUDA_DRV_CHECK(cuModuleGetFunction(&cuFunc, cuMod, "matmul"));

    // Device memory
    size_t bytesLeft   = static_cast<size_t>(rows) * inners * sizeof(float);
    size_t bytesRight  = static_cast<size_t>(inners) * columns * sizeof(float);
    size_t bytesResult = static_cast<size_t>(rows) * columns * sizeof(float);

    CUdeviceptr dLeft, dRight, dResult;
    CUDA_DRV_CHECK(cuMemAlloc(&dLeft, bytesLeft));
    CUDA_DRV_CHECK(cuMemAlloc(&dRight, bytesRight));
    CUDA_DRV_CHECK(cuMemAlloc(&dResult, bytesResult));

    CUDA_DRV_CHECK(cuMemcpyHtoD(dLeft, left,   bytesLeft));
    CUDA_DRV_CHECK(cuMemcpyHtoD(dRight, right, bytesRight));

    void* args[] = { &dLeft, &dRight, &dResult, &rows, &columns, &inners };

    int blockDimX = 16;
    int blockDimY = 16;

    int gridDimX = (columns + blockDimX - 1) / blockDimX;
    int gridDimY = (rows + blockDimY - 1) / blockDimY;

    CUDA_DRV_CHECK(cuLaunchKernel(cuFunc,
                                gridDimX, gridDimY, 1,   // Grid dimension
                                blockDimX, blockDimY, 1, // Block dimension
                                0, 0, args, nullptr));
    CUDA_DRV_CHECK(cuCtxSynchronize());

    CUDA_DRV_CHECK(cuMemcpyDtoH(result, dResult, bytesResult));

    // Cleanup
    CUDA_DRV_CHECK(cuMemFree(dLeft));
    CUDA_DRV_CHECK(cuMemFree(dRight));
    CUDA_DRV_CHECK(cuMemFree(dResult));
    CUDA_DRV_CHECK(cuModuleUnload(cuMod));
    CUDA_DRV_CHECK(cuCtxDestroy(cuCtx));
}

JITCompiler::~JITCompiler() = default;