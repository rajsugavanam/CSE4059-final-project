#include <iostream>
// #include "aabb_old.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "cuda_helper.h"
#include "camera.h"
#include "timer.h"
#include "image_writer.h"

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080

// aabb_old.cuh placed here for testing
class AABB {
    public:
      Vec3 box_min;
      Vec3 box_max;
  
      __host__ __device__ AABB() {
          box_min = Vec3(INFINITY, INFINITY, INFINITY);
          box_max = Vec3(-INFINITY, -INFINITY, -INFINITY);
      }
  
      __host__ __device__ AABB(Vec3 min, Vec3 max) {
          box_min = min;
          box_max = max;
      }
  
      // SOURCE: IRT (p.65)- An Introduction to Ray Tracing, Andrew S. Glassner
      // https://education.siggraph.org/static/HyperGraph/raytrace/rtinter3.htm
      __host__ __device__ bool hitAABB(Ray ray) const {
          // Ray P = O + tD
          Vec3 ray_origin = ray.origin();
          Vec3 ray_direction = ray.direction();
  
          // set t_near and t_far to infty
          float t_near = -INFINITY;
          float t_far = INFINITY;
  
          // for each pair of planes do
          for (int axis = 0; axis < 3; axis++) {
              // if ray is parallel to slab
              if (ray_direction[axis] == 0) {
                  // if ray origin is outside slab
                  if (ray_origin[axis] < box_min[axis] || ray_origin[axis] > box_max[axis]) {
                      return false; // no intersection
                  }
                  continue; // go to next axis to avoid division by zero
              }
  
              // calculate t_near and t_far
              // t0 = (x0 - Ox) / Dx
              float t0 = (box_min[axis] - ray_origin[axis]) / ray_direction[axis];
              float t1 = (box_max[axis] - ray_origin[axis]) / ray_direction[axis];
  
              // swap t0 and t1 if necessary
              if (t0 > t1) {
                  float temp = t0;
                  t0 = t1;
                  t1 = temp;
              }
  
              // update t_near and t_far
              if (t0 > t_near) {
                  t_near = t0;
              }
              if (t1 < t_far) {
                  t_far = t1;
              }
  
              // if the intervals are disjoint or slab is missed
              if (t_near > t_far || t_far < 0) {
                  return false;
              }
          }
          // ray intersects all three slabs
          return true; 
      }
  };


// Modified colorRay function to match include/first_crt.cuh style
__device__ Vec3 colorRay(const Ray& ray, const AABB& box) {
    // Check intersection with box
    if (box.hitAABB(ray)) {
        // If ray hits box, color the pixel red
        return Vec3(1.0f, 0.0f, 0.0f);
    } else {
        // Background color (gradient from blue to white) - exactly like in include/first_crt.cuh
        Vec3 unit_direction = unit_vector(ray.direction());
        float alpha = 0.5f * (unit_direction.y() + 1.0f);  // y = [-1,1] to y = [0,1]
        // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
        return (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
               alpha * Vec3(0.5f, 0.7f, 1.0f);
    }
}

// Rewritten to match exactly the style of rayRender in include/first_crt.cuh
__global__ void renderBoxKernel(Vec3* image_buffer, int width, int height, 
                                Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v, 
                                Vec3 camera_origin) {
    // Calculate pixel coordinates
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        const int pixel_idx = row * width + col;

        // ray params
        const Vec3 pixel_center = pixel00_loc + (col * delta_u) + (row * delta_v);
        const Vec3 ray_direction = pixel_center - camera_origin;

        // Define box with the specified bounds
        // AABB box(Vec3(0.5f, -0.5f, -1.0f), Vec3(-0.5f, 0.5f, -2.0f));
        AABB box(Vec3(-0.5f, -0.5f, -2.0f), Vec3(0.5f, 0.5f, -1.0f));

        Ray ray(camera_origin, ray_direction);
        image_buffer[pixel_idx] = colorRay(ray, box);
    }
}

// Test function
int test_aabb_render() {
    // Allocate host memory for Vec3 buffer (not unsigned char anymore)
    Vec3* h_image = new Vec3[IMAGE_WIDTH * IMAGE_HEIGHT];
    
    // Allocate device memory
    Vec3* d_image;
    cudaMalloc(&d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3));
    
    // Camera and viewport setup - matching rayRender in include/first_crt.cuh exactly
    float aspect_ratio = static_cast<float>(IMAGE_WIDTH) / IMAGE_HEIGHT;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * aspect_ratio;
    float focal_length = 1.0f;
    
    Vec3 camera_origin(0, 0, 0);
    Vec3 horizontal(viewport_width, 0, 0);
    Vec3 vertical(0, -viewport_height, 0);  // Negative to match include/first_crt.cuh
    Vec3 lower_left_corner = camera_origin - horizontal/2 - vertical/2 - Vec3(0, 0, focal_length);
    
    // Calculate pixel delta vectors and top-left pixel - exactly as in include/first_crt.cuh
    Vec3 delta_u = horizontal / static_cast<float>(IMAGE_WIDTH);
    Vec3 delta_v = vertical / static_cast<float>(IMAGE_HEIGHT);
    Vec3 pixel00_loc = lower_left_corner + 0.5f * (delta_u + delta_v);
    
    // Define grid and block dimensions - matching include/first_crt.cuh
    dim3 blocks(16, 16);
    dim3 grid((IMAGE_WIDTH + blocks.x - 1) / blocks.x, 
              (IMAGE_HEIGHT + blocks.y - 1) / blocks.y);
    
    // Launch kernel
    Timer timer;
    timer.start("Rendering AABB");
    renderBoxKernel<<<grid, blocks>>>(d_image, IMAGE_WIDTH, IMAGE_HEIGHT, 
                                      pixel00_loc, delta_u, delta_v, camera_origin);
    cudaDeviceSynchronize();
    timer.stop();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_image, d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3), cudaMemcpyDeviceToHost);
    
    // Save the image using the include/first_crt.cuh approach
    writeToPPM("aabb_render_test.ppm", h_image, IMAGE_WIDTH, IMAGE_HEIGHT);
    
    // Clean up
    delete[] h_image;
    cudaFree(d_image);
    
    std::cout << "AABB render test completed. Output saved to aabb_render_test.ppm" << std::endl;
    return 0;
}

// Main function to be called from test runner
int main(int argc, char** argv) {
    return test_aabb_render();
}
