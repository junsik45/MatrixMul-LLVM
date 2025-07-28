cmake_minimum_required(VERSION 3.10)
project(llvm-matmul-jit)

find_package(CUDA REQUIRED)   # or manually -lcuda
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_library(CUDA_DRIVER_LIBRARY cuda
    HINTS /usr/lib/wsl/lib /usr/local/cuda-12.9/targets/x86_64-linux/lib/stubs
)

if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "libcuda.so not found!")
else()
    message(STATUS "Found libcuda: ${CUDA_DRIVER_LIBRARY}")
endif()



# Use llvm-config to get LLVM directories
execute_process(COMMAND llvm-config --includedir
                OUTPUT_VARIABLE LLVM_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --libdir
                OUTPUT_VARIABLE LLVM_LIB_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --libs core orcjit native mcjit nvptxinfo nvptxdesc nvptxcodegen
                OUTPUT_VARIABLE LLVM_LIBS
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --system-libs
                OUTPUT_VARIABLE LLVM_SYSTEM_LIBS
                OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --cxxflags
                OUTPUT_VARIABLE LLVM_CXXFLAGS
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Add LLVM include directory
include_directories(${LLVM_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/matmul)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")

# Set the source files
set(SOURCES
    src/main.cpp
    src/jit/JITCompiler.cpp
    src/matmul/MatMulKernel.cpp
)

# Create the executable
add_executable(llvm-matmul-jit ${SOURCES})

# Link LLVM libraries
target_link_directories(llvm-matmul-jit PRIVATE ${LLVM_LIB_DIR})
target_link_libraries(llvm-matmul-jit
    ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS}
    ${CUDA_DRIVER_LIBRARY}
)