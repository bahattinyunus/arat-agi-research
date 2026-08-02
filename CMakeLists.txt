cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(AratEdgeInference
    VERSION 1.0.0
    LANGUAGES CXX
)

# Standard C++17/20 compilation settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CUDA if compiler is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compilation enabled. Compiler: ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    message(STATUS "CUDA compiler not found. Custom kernels will compile using CPU fallback path.")
endif()

# Edge Inference Library Sources
set(LIBRARY_SOURCES
    edge_inference/cpp_tensor_runtimes/tensor_runtime.cpp
    edge_inference/memory_allocators/arena_allocator.cpp
    edge_inference/cuda_kernels/activation_kernels.cu
)

# Include paths
include_directories(
    edge_inference/cpp_tensor_runtimes
    edge_inference/memory_allocators
    edge_inference/cuda_kernels
)

# Edge Shared/Static Library Target
add_library(arat_edge_static STATIC ${LIBRARY_SOURCES})

# Target compiler flags
target_compile_features(arat_edge_static PRIVATE cxx_std_17)

if(CMAKE_CUDA_COMPILER)
    # CUDA target settings
    set_target_properties(arat_edge_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Compile demo binary to run test allocations and inference routines
add_executable(edge_demo
    edge_inference/cpp_tensor_runtimes/tensor_runtime.cpp
    edge_inference/memory_allocators/arena_allocator.cpp
    edge_inference/cuda_kernels/activation_kernels.cu
)

target_compile_definitions(edge_demo PRIVATE STANDALONE_DEMO)
target_include_directories(edge_demo PRIVATE
    edge_inference/cpp_tensor_runtimes
    edge_inference/memory_allocators
    edge_inference/cuda_kernels
)
