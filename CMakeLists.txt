# Wow! I'm so evil for requiring 4.0!
cmake_minimum_required(VERSION 4.0)

project(cuda_ray_tracer LANGUAGES CXX CUDA)

# find_package(library)

# include_directories(${library_INCLUDE_DIRS})

# add_subdirectory(src)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS FALSE)

add_executable(crt src/ray_tracer.cu)

target_include_directories(crt PUBLIC src)

# target_link_libraries( ${library_LIBRARIES})

