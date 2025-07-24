#include <iostream>
#include "jit/JITCompiler.h"
#include "utils/MatrixUtils.h"
#define SIZE 128

int main() {
    const int rows = SIZE;    // Set the number of rows
    const int columns = SIZE; // Set the number of columns
    const int inners = SIZE;  // Set the inner dimension

    // Allocate memory for matrices
    float* left   = static_cast<float*>(aligned_alloc(32, rows * inners * sizeof(float)));
    float* right  = static_cast<float*>(aligned_alloc(32, inners * columns * sizeof(float)));
    float* result = static_cast<float*>(aligned_alloc(32, rows * columns * sizeof(float)));

    // Check for allocation success
    if (left == nullptr || right == nullptr || result == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    // Initialize matrices with random values
    initializeMatrix(left, rows, inners);
    initializeMatrix(right, inners, columns);

    // Create JIT compiler instance
    JITCompiler jitCompiler;
    std::string kernelCode = R"(
; LLVM IR code for matrix multiplication kernel goes here
)";

    // Compile and Execute the matrix multiplication kernel
    jitCompiler.initialize();
    jitCompiler.compileAndExecute("", left, right, result, rows, columns, inners);

    // Print the result (optional)
    printMatrix(result, rows, columns);

    // Free allocated memory
    std::free(left);
    std::free(right);
    std::free(result);

    return 0;
}