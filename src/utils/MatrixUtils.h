// MatrixUtils.h
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <cstdlib>
#include <iostream>
#include <ctime>

inline void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(10.0 * std::rand() / RAND_MAX); // Random values [0, 10)
    }
}

inline void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

#endif // MATRIX_UTILS_H