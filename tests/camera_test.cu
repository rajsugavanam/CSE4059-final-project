#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cassert>
#include <iostream>
#include <ostream>
#include <string>

#include "include/crt2.cuh"
// #include "ray.cuh"
#include "camera.h"
#include "obj_reader.cuh"
#include "timer.h"
#include "vec3.cuh"
#include "image_writer.h"

int main() {
    Camera camera(cornell_box_params);

    CUDACameraParams CUDA_camera_params = camera.CUDAparams();
    int pixel_width = CUDA_camera_params.pixel_width;
    int pixel_height = CUDA_camera_params.pixel_height;

    int image_buffer_size = pixel_width * pixel_height;
    size_t image_buffer_byte_size = image_buffer_size * sizeof(Vec3);
    // =====================
    // ===== CONSTANTS =====
    // =====================
    const int block_dim_x = 16;
    const int block_dim_y = 16;
    const int grid_dim_x = (pixel_width + block_dim_x - 1) / block_dim_x;
    const int grid_dim_y = (pixel_height + block_dim_y - 1) / block_dim_y;

    // ===================
    // ===== THE OBJ =====
    // ===================
    Timer timer;
    timer.start("Loading OBJ file");

    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) +
    // "/assets/cubedk.obj");
    ObjReader reader =
        ObjReader(std::string(PROJECT_ROOT) + "/assets/cornell_box/cornell_box_dk.obj");

    reader.readModel();
    Model model = reader.parsedModel;

    assert(!model.modelTriangles.empty());
    timer.stop();

    // ========================
    // ===== MEMORY TRAIN =====
    // ========================
    timer.start("Memory Allocation on GPU");
    // init rgb image output buffer
    Vec3* h_image_buffer{new Vec3[image_buffer_size]};
    Vec3* d_image_buffer;

    // load all the triangles
    Triangle3* h_triangle_mesh = model.modelTriangles.data();
    Triangle3* d_triangle_mesh;
    size_t mesh_size = model.modelTriangles.size();
    size_t triangle_mesh_byte_size = mesh_size * sizeof(Triangle3);
    std::cout << "OBJ MODEL SIZE: " << mesh_size << "\n";

    // from cuRAND
    curandState* d_rng_states = 0;

    int total_threads = block_dim_x * block_dim_y * grid_dim_x * grid_dim_y;
    // malloc state
    cudaMalloc((void**)&d_image_buffer, image_buffer_byte_size);
    cudaMalloc((void**)&d_triangle_mesh, triangle_mesh_byte_size);
    cudaMalloc((void**)&d_rng_states, total_threads * sizeof(curandState));
    cudaMemcpy(d_image_buffer, h_image_buffer, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangle_mesh, h_triangle_mesh, triangle_mesh_byte_size,
               cudaMemcpyHostToDevice);
    timer.stop();

    // ===========================
    // ===== RUNNING ON GPU ======
    // ===========================
    dim3 block_dim(block_dim_x, block_dim_y);
    dim3 grid_dim(grid_dim_x, grid_dim_y);
    timer.start("Init cuRAND");
    initRng<<<grid_dim, block_dim>>>(d_rng_states, pixel_width);
    timer.stop();
    timer.start("Kernel Launching");

    rayRender<<<grid_dim, block_dim>>>(d_image_buffer, d_triangle_mesh,
                                       mesh_size, CUDA_camera_params);
    cudaDeviceSynchronize();
    timer.stop();

    // ============================
    // ===== CPU IMAGE WRITER =====
    // ============================
    timer.start("Copying back to host");
    cudaMemcpy(h_image_buffer, d_image_buffer, image_buffer_byte_size,
               cudaMemcpyDeviceToHost);
    timer.stop();

    timer.start("Outputting to PPM file");
    writeToPPM("camera_test.ppm", h_image_buffer, pixel_width, pixel_height);
    timer.stop();

    // ==========================
    // ===== MEMORY FREEDOM =====
    // ==========================
    delete[] h_image_buffer;
    cudaFree(d_image_buffer);
    cudaFree(d_triangle_mesh);
    cudaFree(d_rng_states);

    return 0;
}
