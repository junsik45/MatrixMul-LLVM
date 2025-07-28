#include "MatMulKernel.h"
#include <iostream>
#include <cstdlib>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormattedStream.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>


    // ------------------------------
    // Helper: declare NVVM sreg readers by name (works across LLVM versions)
    // ------------------------------



static llvm::FunctionCallee getNvvmSReg(llvm::Module &M, llvm::StringRef name) {
    llvm::LLVMContext &C = M.getContext();
    llvm::FunctionType *FT = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(C),     // Return type: i32
        false                          // No parameters
    );
    return M.getOrInsertFunction(name, FT);
}



void MatMulKernel::initializeMatrices(int r, int c, int k) {
    rows = r;
    columns = c;
    inners = k;
    left.resize(rows * inners);
    right.resize(inners * columns);
    result.resize(rows * columns, 0.0f);

    for (int i = 0; i < rows * inners; ++i) {
        left[i] = static_cast<float>(10.0 * std::rand() / RAND_MAX);
    }
    for (int i = 0; i < inners * columns; ++i) {
        right[i] = static_cast<float>(10.0 * std::rand() / RAND_MAX);
    }
}

void MatMulKernel::execute() {
    // Standard matrix multiplication: result = left * right
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < inners; ++k) {
                sum += left[i * inners + k] * right[k * columns + j];
            }
            result[i * columns + j] = sum;
        }
    }
}

void MatMulKernel::printResult() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            std::cout << result[i * columns + j] << " ";
        }
        std::cout << std::endl;
    }
}
llvm::orc::ThreadSafeModule MatMulKernel::createModule(llvm::orc::ThreadSafeContext& TSCtx) {
    using namespace llvm;
    LLVMContext &ctx = *TSCtx.getContext();  // Same code as before

    // ------------------------------
    // Create module & set triple / datalayout
    // ------------------------------
    auto M = std::make_unique<Module>("matmul_module", ctx);
    Triple triple("nvptx64-nvidia-cuda");
    M->setTargetTriple(triple);

    std::string err;
    const Target *T = TargetRegistry::lookupTarget(triple.getTriple(), err);
    if (!T) {
        std::cerr << "Failed to lookup target: " << err << "\n";
        std::exit(1);
    }

    TargetOptions opt;
    std::optional<Reloc::Model> relocModel;
    std::unique_ptr<TargetMachine> TM(
        T->createTargetMachine(triple,
                               /*CPU=*/"sm_89", /*Features=*/"",
                               opt, relocModel));
    if (!TM) {
        std::cerr << "Failed to create TargetMachine!\n";
        std::exit(1);
    }
    M->setDataLayout(TM->createDataLayout());

    // ------------------------------
    // Function signature
    //   void matmul(float* addrspace(1) A,
    //               float* addrspace(1) B,
    //               float* addrspace(1) C,
    //               int rows, int cols, int inner)
    // ------------------------------
    auto &C = ctx;
    auto f32 = Type::getFloatTy(C);
    auto i32 = Type::getInt32Ty(C);
    auto gptrF = PointerType::get(f32, /*addrspace=*/1);

    auto *FT = FunctionType::get(Type::getVoidTy(C),
                                 { gptrF, gptrF, gptrF, i32, i32, i32 },
                                 false);
    auto *F = Function::Create(FT, Function::ExternalLinkage, "matmul", M.get());
    F->setCallingConv(llvm::CallingConv::PTX_Kernel); // <-- THIS IS CRITICAL
    // Mark as kernel
    {
        Metadata *mdVals[] = {
            ValueAsMetadata::get(F),
            MDString::get(C, "kernel"),
            ConstantAsMetadata::get(ConstantInt::get(i32, 1))
        };
        auto *mdNode = MDNode::get(C, mdVals);
        M->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(mdNode);
    }

    // ------------------------------
    // Read arguments
    // ------------------------------
    auto args = F->arg_begin();
    Value *A     = &*args++; A->setName("A");
    Value *B     = &*args++; B->setName("B");
    Value *Cout  = &*args++; Cout->setName("C");
    Value *Rows  = &*args++; Rows->setName("rows");
    Value *Cols  = &*args++; Cols->setName("cols");
    Value *Inner = &*args++; Inner->setName("inner");

    // ------------------------------
    // Blocks
    // ------------------------------
    IRBuilder<> b(C);
    auto *entry   = BasicBlock::Create(C, "entry", F);
    auto *oobRet  = BasicBlock::Create(C, "oob.ret", F);
    auto *kLoop   = BasicBlock::Create(C, "k.loop", F);
    auto *kBody   = BasicBlock::Create(C, "k.body", F);
    auto *kEnd    = BasicBlock::Create(C, "k.end", F);
    auto *retBB   = BasicBlock::Create(C, "ret", F);

    b.SetInsertPoint(entry);

    // ------------------------------
    // Declare NVVM sreg readers
    // ------------------------------
    Value *tid_x    = b.CreateCall(getNvvmSReg(*M, "llvm.nvvm.read.ptx.sreg.tid.x"), {});
    Value *tid_y    = b.CreateCall(getNvvmSReg(*M, "llvm.nvvm.read.ptx.sreg.tid.y"), {});
    Value *blkDim_x = b.CreateCall(getNvvmSReg(*M, "llvm.nvvm.read.ptx.sreg.ntid.x"),  {});
    Value *blkDim_y = b.CreateCall(getNvvmSReg(*M, "llvm.nvvm.read.ptx.sreg.ntid.y"),  {});
    Value *blk_x    = b.CreateCall(getNvvmSReg(*M, "llvm.nvvm.read.ptx.sreg.ctaid.x"), {});
    Value *blk_y    = b.CreateCall(getNvvmSReg(*M, "llvm.nvvm.read.ptx.sreg.ctaid.y"), {});

    // row = blockIdx.y * blockDim.y + threadIdx.y
    // col = blockIdx.x * blockDim.x + threadIdx.x
    Value *row = b.CreateAdd(b.CreateMul(blk_y, blkDim_y), tid_y, "row");
    Value *col = b.CreateAdd(b.CreateMul(blk_x, blkDim_x), tid_x, "col");

    Value *row_ok = b.CreateICmpSLT(row, Rows);
    Value *col_ok = b.CreateICmpSLT(col, Cols);
    Value *inBounds = b.CreateAnd(row_ok, col_ok);
    b.CreateCondBr(inBounds, kLoop, oobRet);

    // ------------------------------
    // k-loop (PHI-based, no allocas)
    // ------------------------------
    b.SetInsertPoint(kLoop);
    auto *kPHI   = b.CreatePHI(i32, 2, "k");
    auto *sumPHI = b.CreatePHI(f32, 2, "sum");
    kPHI->addIncoming(ConstantInt::get(i32, 0), entry);
    sumPHI->addIncoming(ConstantFP::get(f32, 0.0f), entry);

    Value *kCond = b.CreateICmpSLT(kPHI, Inner);
    b.CreateCondBr(kCond, kBody, kEnd);

    // k-body
    b.SetInsertPoint(kBody);
    Value *aIdx = b.CreateAdd(b.CreateMul(row, Inner), kPHI);
    Value *bIdx = b.CreateAdd(b.CreateMul(kPHI, Cols), col);

    Value *aPtr = b.CreateGEP(f32, A, aIdx);
    Value *bPtr = b.CreateGEP(f32, B, bIdx);
    Value *aVal = b.CreateLoad(f32, aPtr);
    Value *bVal = b.CreateLoad(f32, bPtr);

    Value *prod    = b.CreateFMul(aVal, bVal);
    Value *newSum  = b.CreateFAdd(sumPHI, prod);
    Value *kNext   = b.CreateAdd(kPHI, ConstantInt::get(i32, 1));

    kPHI->addIncoming(kNext, kBody);
    sumPHI->addIncoming(newSum, kBody);

    b.CreateBr(kLoop);

    // k-end: store result
    b.SetInsertPoint(kEnd);
    Value *cIdx = b.CreateAdd(b.CreateMul(row, Cols), col);
    Value *cPtr = b.CreateGEP(f32, Cout, cIdx);
    b.CreateStore(sumPHI, cPtr);
    b.CreateBr(retBB);

    // out-of-bounds threads just return
    b.SetInsertPoint(oobRet);
    b.CreateRetVoid();

    // final return
    b.SetInsertPoint(retBB);
    b.CreateRetVoid();

    // ------------------------------
    // Verify
    // ------------------------------
    if (verifyFunction(*F, &errs()) || verifyModule(*M, &errs())) {
        errs() << "<<< Invalid module generated >>>\n";
        M->print(errs(), nullptr);
        std::exit(1);
    }

    return llvm::orc::ThreadSafeModule(std::move(M), llvm::orc::ThreadSafeContext(TSCtx));
}


