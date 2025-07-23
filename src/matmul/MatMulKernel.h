// MatMulKernel.h
#ifndef MATMULKERNEL_H
#define MATMULKERNEL_H

#include <vector>

class MatMulKernel {
public:
    MatMulKernel(int rows, int columns, int inners);
    void initializeMatrices();
    void execute();
    const std::vector<float>& getResult() const;

private:
    int rows_;
    int columns_;
    int inners_;
    std::vector<float> left_;
    std::vector<float> right_;
    std::vector<float> result_;
};

#endif // MATMULKERNEL_H