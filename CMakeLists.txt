# Wow! I'm so evil for requiring 4.0! ... sorry we have to use 3.30
cmake_minimum_required(VERSION 3.30)

project(cuda_ray_tracer LANGUAGES CXX CUDA)

# find_package(library)

# include_directories(${library_INCLUDE_DIRS})

# add_subdirectory(src)

set(PROJECT_NAME crt)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# these make neovim play nice.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS FALSE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# extra compiler flags. we (and by that i mean me, raj) don't like warnings
# so we make them errors.
set(CMAKE_CUDA_FLAGS -Werror=all-warnings)

# LIBRARIES --------------------------
# add_library(cudamath SHARED
#     ${PROJECT_SOURCE_DIR}/src/math/VecN.cu)
add_subdirectory(src/math)
# ------------------------------------

# ------------------------------------
add_executable(${PROJECT_NAME} src/ray_tracer.cu)
target_include_directories(${PROJECT_NAME} PUBLIC src)
# ------------------------------------

# LINK LIBRARIES ---------------------
target_link_libraries(crt PUBLIC cudamath)
# ------------------------------------
