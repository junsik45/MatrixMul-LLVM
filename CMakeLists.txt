cmake_minimum_required(VERSION 3.10)
project(llvm-matmul-jit)

set(CMAKE_CXX_STANDARD 14)

# Find LLVM package
find_package(LLVM REQUIRED CONFIG)

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})

# Set the source files
set(SOURCES
    src/main.cpp
    src/jit/JITCompiler.cpp
    src/matmul/MatMulKernel.cpp
)

# Create the executable
add_executable(llvm-matmul-jit ${SOURCES})

# Link against LLVM libraries
llvm_map_components_to_libnames(llvm_libs core executionengine orcjit native)
target_link_libraries(llvm-matmul-jit ${llvm_libs})