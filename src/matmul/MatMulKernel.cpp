#include "MatMulKernel.h"
#include <iostream>
#include <vector>
#include <cstdlib>

void MatMulKernel::initializeMatrices(int rows, int columns, int inners) {
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
    // Implement the matrix multiplication logic here
    for (size_t i = 0; i < left.size(); ++i) {
        for (size_t j = 0; j < right.size(); ++j) {
            result[i] += left[i] * right[j]; // Simplified for demonstration
        }
    }
}

void MatMulKernel::printResult() const {
    for (const auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}