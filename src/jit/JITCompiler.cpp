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
#include <llvm/ExecutionEngine/Orc/ThreadSafeContext.h>
#include <iostream>

using namespace llvm;

JITCompiler::JITCompiler() {
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
}

void JITCompiler::initialize() {
    auto Context = std::make_unique<ThreadSafeContext>(std::make_unique<LLVMContext>());
    auto JIT = LLJITBuilder().create();
    if (!JIT) {
        std::cerr << "Failed to create JIT: " << toString(JIT.takeError()) << std::endl;
        exit(1);
    }
    JITInstance = std::move(JIT.get());
}

void JITCompiler::compileAndExecute() {
    auto MatMulKernel = std::make_unique<MatMulKernel>();
    auto Module = MatMulKernel->createModule();
    
    if (auto Err = JITInstance->addIRModule(ThreadSafeModule(std::move(Module), std::move(Context)))) {
        std::cerr << "Failed to add module: " << toString(std::move(Err)) << std::endl;
        exit(1);
    }

    auto Func = JITInstance->lookup("matmul");
    if (!Func) {
        std::cerr << "Failed to find function: " << toString(Func.takeError()) << std::endl;
        exit(1);
    }

    using MatMulFuncType = void(*)(float*, float*, float*, int, int, int);
    MatMulFuncType matmulFunc = reinterpret_cast<MatMulFuncType>(Func->getAddress());
    
    // Example usage of the compiled function
    const int SIZE = 512;
    float* left = new float[SIZE * SIZE];
    float* right = new float[SIZE * SIZE];
    float* result = new float[SIZE * SIZE];

    // Initialize matrices and call the function
    matmulFunc(left, right, result, SIZE, SIZE, SIZE);

    // Clean up
    delete[] left;
    delete[] right;
    delete[] result;
}