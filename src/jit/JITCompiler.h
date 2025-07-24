#ifndef JITCOMPILER_H
#define JITCOMPILER_H

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <memory>
#include <string>

class JITCompiler {
public:
    JITCompiler();
    ~JITCompiler();

    void initialize();
    void compileAndExecute(const std::string &kernelCode, float *left, float *right, float *result, int rows, int columns, int inners);

private:
    std::unique_ptr<llvm::orc::LLJIT> jit;
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
};

#endif // JITCOMPILER_H