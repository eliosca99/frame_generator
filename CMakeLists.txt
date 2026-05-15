cmake_minimum_required(VERSION 3.10)
project(frame_generator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)

add_executable(frame_generator  src/main.cpp
                                src/sequential/processing.cpp
                                src/parallel/processing.cu
                                src/utils/png_loader.cpp)

target_include_directories(frame_generator PRIVATE include)

target_link_libraries(frame_generator PRIVATE CUDA::cudart PRIVATE PNG::PNG)
