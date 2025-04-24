# Wow! I'm so evil for requiring 4.0! ... sorry we have to use 3.30
cmake_minimum_required(VERSION 3.30)

project(cuda_ray_tracer LANGUAGES CXX CUDA)

set(PROJECT_NAME crt)

# Device LTO for CUDA i.e. -dlto
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# these make neovim play nice.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES FALSE)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS FALSE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin)

# abs path to proj root so i can run exec file from anywhere
add_compile_definitions(PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# extra compiler flags. we (and by that i mean me, raj) don't like warnings
# so we make them errors.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror=all-warnings")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # key to multiple .cu source files

# INCLUDE DIRECTORIES ------------------------------
include_directories(
  ${PROJECT_SOURCE_DIR}/include
)

# LIBRARIES ------------------------------
add_library(reduction_old STATIC
    src/reduction_old.cu
)
add_library(light_sampling STATIC
    src/light_sampling.cu
)
# If you want to remove global include || target properties
# target_include_directories(reduction PUBLIC include)
#
# set_target_properties(reduction PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

add_library(triangle_mesh STATIC
    src/triangle_mesh.cu
)

# Create the AABB library with optimized settings
add_library(aabb STATIC
    src/aabb.cu
)

# Create Scene Manager library
add_library(scene_manager STATIC
    src/scene_manager.cu
)

target_include_directories(scene_manager PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(scene_manager
    aabb
    triangle_mesh
)

# new reduction library
add_library(reduction STATIC
    src/reduction.cu
)
target_link_libraries(reduction
    aabb
    triangle_mesh
)

# MAIN EXECUTABLE ------------------------------
add_executable(${PROJECT_NAME} "src/ray_tracer.cu")
# Linking libraries
target_link_libraries(${PROJECT_NAME} reduction)

# TESTS ------------------------------
enable_testing()
add_subdirectory(tests)
