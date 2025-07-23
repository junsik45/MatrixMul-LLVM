# LLVM ORC JIT Matrix Multiplication

This project implements a matrix multiplication application using the LLVM ORC JIT (Just-In-Time) compilation framework. The application compiles a matrix multiplication kernel at runtime and executes it, demonstrating the capabilities of LLVM's JIT compilation.

## Project Structure

The project is organized as follows:

```
llvm-matmul-jit
├── src
│   ├── main.cpp               # Entry point of the application
│   ├── jit
│   │   ├── JITCompiler.h      # Header for JITCompiler class
│   │   └── JITCompiler.cpp    # Implementation of JITCompiler class
│   ├── matmul
│   │   ├── MatMulKernel.h     # Header for MatMulKernel class
│   │   └── MatMulKernel.cpp   # Implementation of MatMulKernel class
│   └── utils
│       └── MatrixUtils.h      # Utility functions for matrix operations
├── CMakeLists.txt             # CMake configuration file
└── README.md                   # Project documentation
```

## Building the Project

To build the project, follow these steps:

1. Ensure you have CMake and LLVM installed on your system.
2. Navigate to the project directory:
   ```bash
   cd llvm-matmul-jit
   ```
3. Create a build directory and navigate into it:
   ```bash
   mkdir build && cd build
   ```
4. Run CMake to configure the project:
   ```bash
   cmake ..
   ```
5. Build the project:
   ```bash
   make
   ```

## Running the Application

After building the project, you can run the application using the following command:

```bash
./llvm-matmul-jit
```

## Usage

The application initializes matrices with random values, compiles the matrix multiplication kernel using LLVM's JIT compiler, and executes the kernel to perform matrix multiplication. The results are printed to the console.

## Contributing

Contributions to this project are welcome. Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.