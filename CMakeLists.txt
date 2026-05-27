cmake_minimum_required(VERSION 3.10)
project(frame_generator LANGUAGES CXX)

option(USE_CUDA "Enable CUDA support" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PNG REQUIRED)

set(SOURCES
    src/main.cpp
    src/sequential/processing.cpp
    src/utils/png_utils.cpp
    src/utils/utils.cpp
)

if(USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED True)
        list(APPEND SOURCES src/parallel/processing.cu)
    else()
        message(WARNING "CUDAToolkit not found; building without CUDA support.")
        set(USE_CUDA OFF)
    endif()
endif()

add_executable(frame_generator ${SOURCES})

target_include_directories(frame_generator PRIVATE include)

target_link_libraries(frame_generator PRIVATE PNG::PNG)

if(USE_CUDA)
    target_link_libraries(frame_generator PRIVATE CUDA::cudart)
    target_compile_definitions(frame_generator PRIVATE FRAMEGEN_USE_CUDA)
endif()
