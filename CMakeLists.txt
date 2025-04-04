# Wow! I'm so evil for requiring 4.0!
cmake_minimum_required(VERSION 4.0)
project(cuda_ray_tracer LANGUAGES CXX CUDA)

# find_package(library)

# include_directories(${library_INCLUDE_DIRS})

# add_subdirectory(src)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(crt src/ray_tracer.cu)

# target_link_libraries( ${library_LIBRARIES})

