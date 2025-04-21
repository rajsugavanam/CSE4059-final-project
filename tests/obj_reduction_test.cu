#include <assert.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "reduction.cuh"
#include "cuda_helper.h"
#include "triangle3.cuh"
#include "ray.cuh"
#include "aabb.cuh"
#include "timer.h"
#include "obj_reader.cuh"
#include "image_writer.h"
#include "vec3.cuh"

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

__device__ Vec3 colorRay(const Ray& ray, AABB* box) {
  // Check intersection with box
  if (box->hitAABB(ray)) {
      // If ray hits box, color the pixel red
      return Vec3(1.0f, 0.0f, 0.0f);
  } else {
      // Background color (gradient from blue to white) - exactly like in crt.cuh
      Vec3 unit_direction = unit_vector(ray.direction());
      float alpha = 0.5f * (unit_direction.y() + 1.0f);  // y = [-1,1] to y = [0,1]
      // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
      return (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
             alpha * Vec3(0.5f, 0.7f, 1.0f);
  }
}

// Rewritten to match exactly the style of rayRender in crt.cuh
__global__ void renderBoxKernel(Vec3* image_buffer, int width, int height, 
                              Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v, 
                              Vec3 camera_origin, AABB* box) {
  // Calculate pixel coordinates
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
      const int pixel_idx = row * width + col;

      // ray params
      const Vec3 pixel_center = pixel00_loc + (col * delta_u) + (row * delta_v);
      const Vec3 ray_direction = pixel_center - camera_origin;

      Ray ray(camera_origin, ray_direction);
      image_buffer[pixel_idx] = colorRay(ray, box);
  }
}

// Add a CPU function to compute AABB bounds for comparison
void computeAABBHost(const std::vector<Triangle3>& triangles, Vec3& min_bounds, Vec3& max_bounds) {
    if (triangles.empty()) return;
    
    // Initialize bounds with first vertex of first triangle
    min_bounds = triangles[0].vertex0();
    max_bounds = triangles[0].vertex0();
    
    // Check all vertices of all triangles
    for (const auto& triangle : triangles) {
        // Check vertex 0
        min_bounds = Vec3(
            fminf(min_bounds.x(), triangle.vertex0().x()),
            fminf(min_bounds.y(), triangle.vertex0().y()),
            fminf(min_bounds.z(), triangle.vertex0().z())
        );
        max_bounds = Vec3(
            fmaxf(max_bounds.x(), triangle.vertex0().x()),
            fmaxf(max_bounds.y(), triangle.vertex0().y()),
            fmaxf(max_bounds.z(), triangle.vertex0().z())
        );
        
        // Check vertex 1
        min_bounds = Vec3(
            fminf(min_bounds.x(), triangle.vertex1().x()),
            fminf(min_bounds.y(), triangle.vertex1().y()),
            fminf(min_bounds.z(), triangle.vertex1().z())
        );
        max_bounds = Vec3(
            fmaxf(max_bounds.x(), triangle.vertex1().x()),
            fmaxf(max_bounds.y(), triangle.vertex1().y()),
            fmaxf(max_bounds.z(), triangle.vertex1().z())
        );
        
        // Check vertex 2
        min_bounds = Vec3(
            fminf(min_bounds.x(), triangle.vertex2().x()),
            fminf(min_bounds.y(), triangle.vertex2().y()),
            fminf(min_bounds.z(), triangle.vertex2().z())
        );
        max_bounds = Vec3(
            fmaxf(max_bounds.x(), triangle.vertex2().x()),
            fmaxf(max_bounds.y(), triangle.vertex2().y()),
            fmaxf(max_bounds.z(), triangle.vertex2().z())
        );
    }
}

int main () {
    // ===================
    // ===== THE OBJ =====
    // ===================
    Timer timer;
    timer.start("Loading OBJ file");
    ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/dk.obj");
    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/large_sphere.obj");
    reader.readModel();
    Model model = reader.parsedModel;
    assert(!model.modelTriangles.empty());
    timer.stop();


    Vec3* image_buffer_h{new Vec3[image_buffer_size]};
    Vec3* image_buffer_d;

    // load all the triangles
    Triangle3* triangles_h = model.modelTriangles.data();
    Triangle3* triangles_d;
    size_t mesh_size = model.modelTriangles.size();
    size_t mesh_byte_size = mesh_size * sizeof(Triangle3);
    std::cout << "OBJ MODEL SIZE: " << mesh_size << "\n";

    // malloc state
    cudaMalloc((void**)&image_buffer_d, image_buffer_byte_size);
    cudaMalloc((void**)&triangles_d, mesh_byte_size);
    cudaMemcpy(image_buffer_d, image_buffer_h, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(triangles_d, triangles_h, mesh_byte_size, cudaMemcpyHostToDevice);
    timer.stop();

    // ===========================
    // ===== RUNNING ON GPU ======
    // ===========================

    // INIT AABB
    AABB* d_aabb = nullptr;
    AABB* h_aabb = new AABB[1]{};
    size_t aabb_size = sizeof(AABB);

    // Initialize with extreme values to ensure proper min/max calculation
    float* h_aabb_min = new float[3]{INFINITY, INFINITY, INFINITY};
    float* h_aabb_max = new float[3]{-INFINITY, -INFINITY, -INFINITY};
    float* d_aabb_min = nullptr;
    float* d_aabb_max = nullptr;

    cudaMalloc((void**)&d_aabb, aabb_size);
    cudaMalloc((void**)&d_aabb_min, 3 * sizeof(float));
    cudaMalloc((void**)&d_aabb_max, 3 * sizeof(float));
    cudaMemcpy(d_aabb_min, h_aabb_min, 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_aabb_max, h_aabb_max, 3 * sizeof(float),
                cudaMemcpyHostToDevice);

    // REDUCTION
    dim3 aabb_block_size(256);
    dim3 aabb_grid_size((mesh_size + aabb_block_size.x - 1) / aabb_block_size.x);
    timer.start("AABB reduction");

    // Allocate enough shared memory for 3 arrays of floats (min/max x, y, z)
    size_t sharedMemSize = 3 * aabb_block_size.x * sizeof(float);
    
    minTriangleMesh<<<aabb_grid_size, aabb_block_size, sharedMemSize>>>(
        triangles_d, mesh_size, d_aabb_min, INFINITY);
    maxTriangleMesh<<<aabb_grid_size, aabb_block_size, sharedMemSize>>>(
        triangles_d, mesh_size, d_aabb_max, -INFINITY);
    cudaDeviceSynchronize();
    timer.stop();

    // copy back to host
    cudaMemcpy(h_aabb_min, d_aabb_min, 3 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aabb_max, d_aabb_max, 3 * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute AABB CPU Version
    timer.start("CPU AABB computation");
    Vec3 cpu_min_bounds, cpu_max_bounds;
    computeAABBHost(model.modelTriangles, cpu_min_bounds, cpu_max_bounds);
    timer.stop();
  
    // CPU vs GPU AABB comparison
    bool same_aabb = false;
    float epsilon = 1e-5f;
    
    if (std::abs(h_aabb_min[0] - cpu_min_bounds.x()) > epsilon || 
        std::abs(h_aabb_min[1] - cpu_min_bounds.y()) > epsilon || 
        std::abs(h_aabb_min[2] - cpu_min_bounds.z()) > epsilon ||
        std::abs(h_aabb_max[0] - cpu_max_bounds.x()) > epsilon || 
        std::abs(h_aabb_max[1] - cpu_max_bounds.y()) > epsilon || 
        std::abs(h_aabb_max[2] - cpu_max_bounds.z()) > epsilon) {
        same_aabb = true;
    }

    // Copy AABB data to device
    h_aabb[0].box_min = Vec3(h_aabb_min[0], h_aabb_min[1], h_aabb_min[2]);
    h_aabb[0].box_max = Vec3(h_aabb_max[0], h_aabb_max[1], h_aabb_max[2]);
    cudaMemcpy(d_aabb, h_aabb, aabb_size, cudaMemcpyHostToDevice);
    
    dim3 block_size(16, 16);
    dim3 grid_size((pixel_width + block_size.x - 1) / block_size.x,
                   (pixel_height + block_size.y - 1) / block_size.y);

    // ======================
    // ===== RENDERING ======
    // ======================
    timer.start("Ray rendering");
    renderBoxKernel<<<grid_size, block_size>>>(
        image_buffer_d, pixel_width, pixel_height, pixel00_loc, pixel_delta_u,
        pixel_delta_v, camera_center, d_aabb);
    
    cudaDeviceSynchronize();
    timer.stop();

    // ============================
    // ===== CPU IMAGE WRITER =====
    // ============================
    timer.start("Copying back to host");
    cudaMemcpy(image_buffer_h, image_buffer_d, image_buffer_byte_size,
               cudaMemcpyDeviceToHost);
    timer.stop();

    timer.start("Outputting to PPM file");
    writeToPPM("obj_reduction_test.ppm", image_buffer_h, pixel_width, pixel_height);
    timer.stop();

    // print AABB from GPU reduction
    std::cout << "GPU AABB min: " << h_aabb_min[0] << ", " << h_aabb_min[1] << ", "
    << h_aabb_min[2] << "\n";
    std::cout << "GPU AABB max: " << h_aabb_max[0] << ", " << h_aabb_max[1] << ", "
    << h_aabb_max[2] << "\n";

    // Print CPU AABB results
    std::cout << "CPU AABB min: " << cpu_min_bounds.x() << ", " 
              << cpu_min_bounds.y() << ", " << cpu_min_bounds.z() << "\n";
    std::cout << "CPU AABB max: " << cpu_max_bounds.x() << ", " 
              << cpu_max_bounds.y() << ", " << cpu_max_bounds.z() << "\n";
              

    // ==========================
    // ===== MEMORY FREEDOM =====
    // ==========================
    delete[] image_buffer_h;
    delete[] h_aabb;
    delete[] h_aabb_min;
    delete[] h_aabb_max;
    cudaFree(image_buffer_d);
    cudaFree(triangles_d);
    cudaFree(d_aabb);
    cudaFree(d_aabb_min);
    cudaFree(d_aabb_max);

    if (same_aabb) {
        std::cerr << "AABB reduction test failed!" << std::endl;
        return 1;
    } else {
        std::cout << "AABB reduction test passed!" << std::endl;
        return 0;
    }
}