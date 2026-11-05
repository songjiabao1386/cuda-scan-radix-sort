cmake_minimum_required(VERSION 3.18)
project(GpuScanRadixSort CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gpu_test 
    src/main.cu
)

set_target_properties(gpu_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
