#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <math/VecN.cuh>
// #include <math/vec3.cuh>
#include <ostream>

#include "vec3.cuh"
#include "timer.h"

#define TEST_SIZE 4
const int pixel_width = 1920;
const int pixel_height = 1080;
const int image_buffer_size = pixel_width * pixel_height;
size_t image_buffer_byte_size = image_buffer_size * sizeof(Vec3);

__global__ void GreenRedRender(Vec3* image_buffer, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int pixel_idx = (row * width + col);  // 3 channels (RGB)
        // image_buffer[pixel_idx] = float(col) / float(width);       // Red
        // image_buffer[pixel_idx + 1] = float(row) / float(height);  // Green
        // image_buffer[pixel_idx + 2] = 0.0f;                        // Blue
        image_buffer[pixel_idx] =
            Vec3(float(col) / float(width), float(row) / float(height), 0.0f);
    }
}

// Fucntion declaserioansen
void WriteToPPM(const char* filename, Vec3* image_buffer, int width,
                int height);
int main() {
    Timer timer;
    float vec1_arr[TEST_SIZE] = {1.0, 2.0, 3.0, 4.0};
    float vec2_arr[TEST_SIZE] = {7.0, 3.0, -1.0, 8.0};

    VecN<float>* float_vec1 = new VecN<float>(TEST_SIZE, vec1_arr);
    VecN<float>* float_vec2 = new VecN<float>(TEST_SIZE, vec2_arr);
    VecN<float>* result_vec = float_vec1->deviceAdd(float_vec2);

    std::cout << "Addition:" << std::endl;
    int N = result_vec->N;
    float* contents = result_vec->pv;
    for (int i = 0; i < N; i++) {
        std::cout << i << ": " << contents[i] << std::endl;
    }

    std::cout << "Dot Product:" << std::endl;
    std::cout << float_vec1->deviceDot(float_vec2) << std::endl;

    // cuda stuff
    timer.start("Memory Allocation on GPU");
    Vec3* image_buffer_h = new Vec3[image_buffer_size]();
    Vec3* image_buffer_d;
    std::cout << "Vec3 size: " << sizeof(Vec3) << std::endl;
    // float* image_buffer_h{new float[image_buffer_size * 3]{}};
    // float* image_buffer_d;
    cudaMalloc((void**)&image_buffer_d, image_buffer_byte_size);
    cudaMemcpy(image_buffer_d, image_buffer_h, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    timer.stop();

    timer.start("Kernel Launching");
    dim3 block_size(16, 16);
    dim3 grid_size((pixel_width + block_size.x - 1) / block_size.x,
                   (pixel_height + block_size.y - 1) / block_size.y);
    GreenRedRender<<<grid_size, block_size>>>(image_buffer_d, pixel_width,
                                              pixel_height);
    cudaDeviceSynchronize();
    timer.stop();

    timer.start("Copying back to host");
    cudaMemcpy(image_buffer_h, image_buffer_d, image_buffer_byte_size,
               cudaMemcpyDeviceToHost);
    timer.stop();

    timer.start("Outputting to PPM file");
    WriteToPPM("outout.ppm", image_buffer_h, pixel_width, pixel_height);
    timer.stop();

    // free mem
    delete[] image_buffer_h;
    cudaFree(image_buffer_d);

    return 0;
}

void WriteToPPM(const char* filename, Vec3* image_buffer, int pixel_width,
                int pixel_height) {
    std::ofstream os(filename);
    os << "P3\n" << pixel_width << " " << pixel_height << "\n255\n";
    for (int j = 0; j < pixel_height; j++) {
        for (int i = 0; i < pixel_width; i++) {
            int pixel_idx = (j * pixel_width + i);
            // int r = static_cast<int>(image_buffer[pixel_idx] * 255.999);
            // int g = static_cast<int>(image_buffer[pixel_idx + 1] * 255.999);
            // int b = static_cast<int>(image_buffer[pixel_idx + 2] * 255.999);
            int r = static_cast<int>(image_buffer[pixel_idx].x() * 255.999);
            int g = static_cast<int>(image_buffer[pixel_idx].y() * 255.999);
            int b = static_cast<int>(image_buffer[pixel_idx].z() * 255.999);
            os << r << " " << g << " " << b << "\n";
        }
    }
    os.close();
}
