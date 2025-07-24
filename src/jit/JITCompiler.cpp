#include "JITCompiler.h"
#include "MatMulKernel.h"
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <iostream>

using namespace llvm;

JITCompiler::JITCompiler() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
}

void JITCompiler::initialize() {
    auto Context = std::make_unique<llvm::orc::ThreadSafeContext>(std::make_unique<llvm::LLVMContext>());
    auto JIT = llvm::orc::LLJITBuilder().create();
    if (!JIT) {
        std::cerr << "Failed to create JIT: " << toString(JIT.takeError()) << std::endl;
        exit(1);
    }
    jit = std::move(JIT.get());
}

void JITCompiler::compileAndExecute(const std::string &kernelCode, float *left, float *right, float *result, int rows, int columns, int inners) {

    auto matMulKernel = std::make_unique<MatMulKernel>();
    auto TSM = matMulKernel->createModule();

    // First, create the actual LLVMContext and wrap it
    auto ctx = std::make_unique<llvm::LLVMContext>();
    llvm::orc::ThreadSafeContext tsContext(std::move(ctx));

    // // Now construct the ThreadSafeModule with a reference to tsContext
    // llvm::orc::ThreadSafeModule TSM(std::move(Module), tsContext);

    if (auto Err = jit->addIRModule(std::move(TSM))) {
        std::cerr << "Failed to add module: " << toString(std::move(Err)) << std::endl;
        exit(1);
    }    
    auto Func = jit->lookup("matmul");
    if (!Func) {
        std::cerr << "Failed to find function: " << toString(Func.takeError()) << std::endl;
        exit(1);
    }

    using MatMulFuncType = void(*)(float*, float*, float*, int, int, int);
    MatMulFuncType matmulFunc = reinterpret_cast<MatMulFuncType>(Func->getValue());
    
    matmulFunc(left, right, result, rows, columns, inners);
}

JITCompiler::~JITCompiler() = default;