// MatMulKernel.h
#pragma once

#include <vector>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <memory>

class MatMulKernel {
public:
    void initializeMatrices(int rows, int columns, int inners);
    void execute();
    void printResult() const;
    llvm::orc::ThreadSafeModule createModule(llvm::orc::ThreadSafeContext& TSCtx);



private:
    int rows = 0, columns = 0, inners = 0;
    std::vector<float> left, right, result;
};