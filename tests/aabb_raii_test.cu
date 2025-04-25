#include <iostream>
#include "aabb.cuh"
#include "ray.cuh"
#include "vec3.cuh"
// #include "cuda_helper.h"
#include "ray_color.cuh"
#include "camera.h"
#include "timer.h"
#include "image_writer.h"

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080

__device__ Vec3 colorRay(const Ray& ray, const AABB* box) {

    if (box->hitAABB(ray, 0)) {
        // If ray hits box, color the pixel red
        return Vec3(1.0f, 0.0f, 0.0f);
    } else {
        // Background color (gradient from blue to white)
        return sky_bg(ray);
    }
}

// Rewritten kernel to use the AABB class
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

// Test function
int test_aabb_render() {
    // Allocate host memory for Vec3 buffer
    Vec3* h_image = new Vec3[IMAGE_WIDTH * IMAGE_HEIGHT];
    
    // Allocate device memory
    Vec3* d_image;
    cudaMalloc(&d_image, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(Vec3));
    
    // Using camera class from crt.cuh
    CameraParams params;
    Camera camera(params);
    CUDACameraParams camera_params = camera.CUDAparams();
    Vec3 pixel00_loc = camera_params.pixel00_loc;
    Vec3 delta_u = camera_params.pixel_delta_u;
    Vec3 delta_v = camera_params.pixel_delta_v;
    Vec3 camera_origin = camera_params.center;

    
    // Create and initialize an AABB with 1 object
    AABB box(1);
    
    // Set box dimensions - min point at (-0.5, -0.5, -2.0) and max at (0.5, 0.5, -1.0)
    box.h_minx[0] = -0.5f;
    box.h_miny[0] = -0.5f;
    box.h_minz[0] = -2.0f;
    box.h_maxx[0] = 0.5f;
    box.h_maxy[0] = 0.5f;
    box.h_maxz[0] = -1.0f;
    
    // Copy data to device
    box.AABBMemcpyHtD();
    
    // Define grid and block dimensions - matching crt.cuh
    dim3 blocks(16, 16);
    dim3 grid((IMAGE_WIDTH + blocks.x - 1) / blocks.x, 
              (IMAGE_HEIGHT + blocks.y - 1) / blocks.y);
    
    // Create device copy of AABB object
    AABB* d_box;
    cudaMalloc(&d_box, sizeof(AABB));
    cudaMemcpy(d_box, &box, sizeof(AABB), cudaMemcpyHostToDevice);
    
    // Launch kernel
    Timer timer;
    timer.start("Rendering AABB RAII");
    renderBoxKernel<<<grid, blocks>>>(d_image, IMAGE_WIDTH, IMAGE_HEIGHT, 
                                      pixel00_loc, delta_u, delta_v, camera_origin, d_box);
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
    
    // Save the image using the crt.cuh approach
    writeToPPM("aabb_raii_test.ppm", h_image, IMAGE_WIDTH, IMAGE_HEIGHT);
    
    // Clean up
    delete[] h_image;
    cudaFree(d_image);
    cudaFree(d_box);
    
    std::cout << "AABB render test completed. Output saved to aabb_raii_test.ppm" << std::endl;
    return 0;
}

// Main function to be called from test runner
int main(int argc, char** argv) {
    return test_aabb_render();
}
