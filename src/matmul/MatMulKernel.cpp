#include "MatMulKernel.h"
#include <iostream>
#include <cstdlib>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
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
llvm::orc::ThreadSafeModule MatMulKernel::createModule() {
    auto context = std::make_unique<llvm::LLVMContext>();
    auto& ctx = *context;  // Get a reference for easier use
    auto module = std::make_unique<llvm::Module>("matmul_module", ctx);

    // Define function signature: void matmul(float*, float*, float*, int, int, int)
    auto floatPtrTy = llvm::Type::getFloatTy(ctx)->getPointerTo();
    auto intTy = llvm::Type::getInt32Ty(ctx);
    std::vector<llvm::Type*> params = {floatPtrTy, floatPtrTy, floatPtrTy, intTy, intTy, intTy};
    auto funcType = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), params, false);

    auto func = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, "matmul", module.get());

    auto args = func->args().begin();
    llvm::Value* leftPtr = args++;
    llvm::Value* rightPtr = args++;
    llvm::Value* resultPtr = args++;
    llvm::Value* rows = args++;
    llvm::Value* columns = args++;
    llvm::Value* inners = args++;

    llvm::IRBuilder<> builder(ctx);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", func);
    builder.SetInsertPoint(entry);

    // Outer loop: for (i = 0; i < rows; ++i)
    llvm::Value* i = builder.CreateAlloca(intTy, nullptr, "i");
    builder.CreateStore(llvm::ConstantInt::get(intTy, 0), i);

    llvm::BasicBlock* loop_i_cond = llvm::BasicBlock::Create(ctx, "loop_i_cond", func);
    llvm::BasicBlock* loop_i_body = llvm::BasicBlock::Create(ctx, "loop_i_body", func);
    llvm::BasicBlock* loop_i_end = llvm::BasicBlock::Create(ctx, "loop_i_end", func);
    builder.CreateBr(loop_i_cond);

    builder.SetInsertPoint(loop_i_cond);
    llvm::Value* i_val = builder.CreateLoad(intTy, i);
    llvm::Value* i_cond = builder.CreateICmpSLT(i_val, rows);
    builder.CreateCondBr(i_cond, loop_i_body, loop_i_end);

    builder.SetInsertPoint(loop_i_body);

    // j loop
    llvm::Value* j = builder.CreateAlloca(intTy, nullptr, "j");
    builder.CreateStore(llvm::ConstantInt::get(intTy, 0), j);

    llvm::BasicBlock* loop_j_cond = llvm::BasicBlock::Create(ctx, "loop_j_cond", func);
    llvm::BasicBlock* loop_j_body = llvm::BasicBlock::Create(ctx, "loop_j_body", func);
    llvm::BasicBlock* loop_j_end = llvm::BasicBlock::Create(ctx, "loop_j_end", func);
    builder.CreateBr(loop_j_cond);

    builder.SetInsertPoint(loop_j_cond);
    llvm::Value* j_val = builder.CreateLoad(intTy, j);
    llvm::Value* j_cond = builder.CreateICmpSLT(j_val, columns);
    builder.CreateCondBr(j_cond, loop_j_body, loop_j_end);

    builder.SetInsertPoint(loop_j_body);

    llvm::Value* sum = builder.CreateAlloca(llvm::Type::getFloatTy(ctx), nullptr, "sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx, llvm::APFloat(0.0f)), sum);

    // k loop
    llvm::Value* k = builder.CreateAlloca(intTy, nullptr, "k");
    builder.CreateStore(llvm::ConstantInt::get(intTy, 0), k);

    llvm::BasicBlock* loop_k_cond = llvm::BasicBlock::Create(ctx, "loop_k_cond", func);
    llvm::BasicBlock* loop_k_body = llvm::BasicBlock::Create(ctx, "loop_k_body", func);
    llvm::BasicBlock* loop_k_end = llvm::BasicBlock::Create(ctx, "loop_k_end", func);
    builder.CreateBr(loop_k_cond);

    builder.SetInsertPoint(loop_k_cond);
    llvm::Value* k_val = builder.CreateLoad(intTy, k);
    llvm::Value* k_cond = builder.CreateICmpSLT(k_val, inners);
    builder.CreateCondBr(k_cond, loop_k_body, loop_k_end);

    builder.SetInsertPoint(loop_k_body);
    // Compute left[i * inners + k]
    llvm::Value* i_inner = builder.CreateMul(i_val, inners);
    llvm::Value* left_idx = builder.CreateAdd(i_inner, k_val);
    llvm::Value* left_elem_ptr = builder.CreateGEP(llvm::Type::getFloatTy(ctx), leftPtr, left_idx);
    llvm::Value* left_val = builder.CreateLoad(llvm::Type::getFloatTy(ctx), left_elem_ptr);

    // Compute right[k * columns + j]
    llvm::Value* k_col = builder.CreateMul(k_val, columns);
    llvm::Value* right_idx = builder.CreateAdd(k_col, j_val);
    llvm::Value* right_elem_ptr = builder.CreateGEP(llvm::Type::getFloatTy(ctx), rightPtr, right_idx);
    llvm::Value* right_val = builder.CreateLoad(llvm::Type::getFloatTy(ctx), right_elem_ptr);

    // Multiply and add to sum
    llvm::Value* prod = builder.CreateFMul(left_val, right_val);
    llvm::Value* sum_val = builder.CreateLoad(llvm::Type::getFloatTy(ctx), sum);
    llvm::Value* new_sum = builder.CreateFAdd(sum_val, prod);
    builder.CreateStore(new_sum, sum);

    // k++
    llvm::Value* k_next = builder.CreateAdd(k_val, llvm::ConstantInt::get(intTy, 1));
    builder.CreateStore(k_next, k);
    builder.CreateBr(loop_k_cond);

    builder.SetInsertPoint(loop_k_end);

    // result[i * columns + j] = sum
    llvm::Value* i_col = builder.CreateMul(i_val, columns);
    llvm::Value* res_idx = builder.CreateAdd(i_col, j_val);
    llvm::Value* res_ptr = builder.CreateGEP(llvm::Type::getFloatTy(ctx), resultPtr, res_idx);
    llvm::Value* final_sum = builder.CreateLoad(llvm::Type::getFloatTy(ctx), sum);
    builder.CreateStore(final_sum, res_ptr);

    // j++
    llvm::Value* j_next = builder.CreateAdd(j_val, llvm::ConstantInt::get(intTy, 1));
    builder.CreateStore(j_next, j);
    builder.CreateBr(loop_j_cond);

    builder.SetInsertPoint(loop_j_end);

    // i++
    llvm::Value* i_next = builder.CreateAdd(i_val, llvm::ConstantInt::get(intTy, 1));
    builder.CreateStore(i_next, i);
    builder.CreateBr(loop_i_cond);

    builder.SetInsertPoint(loop_i_end);
    builder.CreateRetVoid();

    // Validate IR
    llvm::verifyFunction(*func);
    llvm::verifyModule(*module);
    module->print(llvm::errs(), nullptr);
    return llvm::orc::ThreadSafeModule(std::move(module), std::move(context));
}
