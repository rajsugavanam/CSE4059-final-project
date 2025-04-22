#include <iostream>
#include <fstream>
#include <vector>
#include "aabb.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "cuda_helper.h"
#include "camera.h"

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080

// Modified colorRay function to match crt.cuh style
__device__ Vec3 colorRay(const Ray& ray, const AABB& box) {
    // Check intersection with box
    if (box.hitAABB(ray)) {
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
    
    // Camera and viewport setup - matching rayRender in crt.cuh exactly
    float aspect_ratio = static_cast<float>(IMAGE_WIDTH) / IMAGE_HEIGHT;
    float viewport_height = 2.0f;
    float viewport_width = viewport_height * aspect_ratio;
    float focal_length = 1.0f;
    
    Vec3 camera_origin(0, 0, 0);
    Vec3 horizontal(viewport_width, 0, 0);
    Vec3 vertical(0, -viewport_height, 0);  // Negative to match crt.cuh
    Vec3 lower_left_corner = camera_origin - horizontal/2 - vertical/2 - Vec3(0, 0, focal_length);
    
    // Calculate pixel delta vectors and top-left pixel - exactly as in crt.cuh
    Vec3 delta_u = horizontal / static_cast<float>(IMAGE_WIDTH);
    Vec3 delta_v = vertical / static_cast<float>(IMAGE_HEIGHT);
    Vec3 pixel00_loc = lower_left_corner + 0.5f * (delta_u + delta_v);
    
    // Define grid and block dimensions - matching crt.cuh
    dim3 blocks(16, 16);
    dim3 grid((IMAGE_WIDTH + blocks.x - 1) / blocks.x, 
              (IMAGE_HEIGHT + blocks.y - 1) / blocks.y);
    
    // Launch kernel
    renderBoxKernel<<<grid, blocks>>>(d_image, IMAGE_WIDTH, IMAGE_HEIGHT, 
                                      pixel00_loc, delta_u, delta_v, camera_origin);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_image, d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3), cudaMemcpyDeviceToHost);
    
    // Save the image using the crt.cuh approach
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
