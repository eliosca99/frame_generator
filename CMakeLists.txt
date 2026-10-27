cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "Host compiler for NVCC" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12" CACHE FILEPATH "CXX compiler" FORCE)
set(CMAKE_C_COMPILER "/usr/bin/gcc-12" CACHE FILEPATH "C compiler" FORCE)
project(frame_generator LANGUAGES CXX)

option(USE_CUDA "Enable CUDA support" ON)

set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PNG REQUIRED)

set(SOURCES
    src/main.cpp
    src/sequential/interpolation.cpp
    src/sequential/block_matching.cpp
    src/utils/png_utils.cpp
    src/utils/utils.cpp
)

if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
        if(NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR "CUDAToolkit found but no CUDA compiler is available.")
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED True)
        list(APPEND SOURCES src/parallel/interpolation_cuda.cu)
    else()
        message(WARNING "CUDAToolkit not found; building without CUDA support.")
        set(USE_CUDA OFF)
    endif()
endif()

add_executable(frame_generator ${SOURCES}
    src/parallel/block_matching_cuda.cu)

target_include_directories(frame_generator PRIVATE include)

target_link_libraries(frame_generator PRIVATE PNG::PNG)

if(USE_CUDA)
    target_link_libraries(frame_generator PRIVATE CUDA::cudart)
    target_compile_definitions(frame_generator PRIVATE FRAMEGEN_USE_CUDA)
endif()
