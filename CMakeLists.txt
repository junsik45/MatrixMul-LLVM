cmake_minimum_required(VERSION 3.10)
project(llvm-matmul-jit)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Find LLVM package
find_package(LLVM REQUIRED CONFIG)

message(STATUS "LLVM include dirs: ${LLVM_INCLUDE_DIRS}")

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/src/matmul) # <-- Add this line


# Set the source files
set(SOURCES
    src/main.cpp
    src/jit/JITCompiler.cpp
    src/matmul/MatMulKernel.cpp
)

# Create the executable
add_executable(llvm-matmul-jit ${SOURCES})

# Link against LLVM libraries
llvm_map_components_to_libnames(llvm_libs core executionengine orcjit native mcjit)
target_link_libraries(llvm-matmul-jit ${llvm_libs})