#include <cuda_runtime.h>

#include <iostream>
#include <math/VecN.cuh>
#include <ostream>

#include "timer.h"

#define TEST_SIZE 4
const int pixel_width = 1920;
const int pixel_height = 1080;
const int image_buffer_size = pixel_width * pixel_height;
size_t image_buffer_byte_size = image_buffer_size * sizeof(float) * 3;

__global__ void GreenRedRender(float* image_buffer, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int pixel_idx = (row * width + col) * 3;  // 3 channels (RGB)
        image_buffer[pixel_idx] = float(col) / float(width);       // Red
        image_buffer[pixel_idx + 1] = float(row) / float(height);  // Green
        image_buffer[pixel_idx + 2] = 0.0f;                        // Blue
    }
}

// Fucntion declaserioansen
void WriteToPPM(const char* filename, float* image_buffer, int width,
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
    float* image_buffer_h{new float[image_buffer_size * 3]{}};
    float* image_buffer_d;
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

    return 0;
}

void WriteToPPM(const char* filename, float* image_buffer, int pixel_width,
                int pixel_height) {
    std::cout << "P3\n" << pixel_width << " " << pixel_height << "\n255\n";
    for (int j = 0; j < pixel_height; j++) {
        for (int i = 0; i < pixel_width; i++) {
            int pixel_idx = (j * pixel_width + i) * 3;
            int r = static_cast<int>(image_buffer[pixel_idx] * 255.999);
        }
    }
}