#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "crt.cuh"
// #include "ray.cuh"
#include "timer.h"
#include "vec3.cuh"
#include "obj_reader.cuh"
#include "image_writer.h"

// CUDA Error Handling
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(error) << std::endl;                 \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// perhaps move all this to a struct...
const float aspect_ratio = 16.0f / 9.0f;
const int pixel_height = 1080;
const int pixel_width = static_cast<int>(pixel_height * aspect_ratio);
const float focal_length = 1.0f;
const float viewport_height = 2.0f;
const float viewport_width =
    viewport_height * (float(pixel_width) / pixel_height);
const float camera_x = 0.0f;
const float camera_y = 0.0f;
const float camera_z = 0.0f;
// const Vec3 camera_center = Vec3(0.0f, -6.0f, 7.0f); // the sphere is at a weird location
const Vec3 camera_center = Vec3(camera_x, camera_y, camera_z);

// viewport vector
const Vec3 viewport_u = Vec3(viewport_width, 0.0f, 0.0f);
const Vec3 viewport_v = Vec3(0.0f, -viewport_height, 0.0f);

const Vec3 pixel_delta_u = viewport_u / pixel_width;
const Vec3 pixel_delta_v = viewport_v / pixel_height;

const Vec3 viewport_upper_left = camera_center -
                                 Vec3(0.0f, 0.0f, focal_length) -
                                 viewport_u / 2 - viewport_v / 2;
const Vec3 pixel00_loc =
    viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
const int image_buffer_size = pixel_width * pixel_height;
const size_t image_buffer_byte_size = image_buffer_size * sizeof(Vec3);

int main() {
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
    
    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/TyDonkeyKR.obj");
    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/large_sphere.obj");
    ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/cubedk.obj");
    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/dk.obj");
    
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
    curandState *d_rng_states = 0;

    int total_threads = block_dim_x * block_dim_y * grid_dim_x * grid_dim_y;
    // malloc state
    cudaMalloc((void**)&d_image_buffer, image_buffer_byte_size);
    cudaMalloc((void**)&d_triangle_mesh, triangle_mesh_byte_size);
    cudaMalloc((void**)&d_rng_states, total_threads * sizeof(curandState));
    cudaMemcpy(d_image_buffer, h_image_buffer, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangle_mesh, h_triangle_mesh, triangle_mesh_byte_size, cudaMemcpyHostToDevice);
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
    
    rayRender<<<grid_dim, block_dim>>>(
        d_image_buffer, pixel_width, pixel_height, pixel00_loc, pixel_delta_u,
        pixel_delta_v, camera_center, d_triangle_mesh, mesh_size);
    
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
    writeToPPM("outout.ppm", h_image_buffer, pixel_width, pixel_height);
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