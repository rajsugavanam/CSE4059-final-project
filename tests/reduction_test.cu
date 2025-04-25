#include <vector>
#include <random>
#include <algorithm>
// #include <numeric>

#include <cuda_runtime.h>
#include <cuda/std/limits>
#include "reduction_old.cuh"
#include "cuda_helper.h"
#include "timer.h"

static constexpr int BLOCK_SIZE = 1024;

// Calculate CPU results for comparison
float calculateMinCPU(const std::vector<float>& data) {
    return *std::min_element(data.begin(), data.end());
}

float calculateMaxCPU(const std::vector<float>& data) {
    return *std::max_element(data.begin(), data.end());
}


float reductionMin(const float* d_input, int size) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    float identity = INFINITY;
    
    // Allocate and initialize output value with identity
    float* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_result, &identity, sizeof(float), cudaMemcpyHostToDevice);
    
    Timer timer;
    timer.start("GPU Minimum Calculation");
    
    // Launch kernel with error checking
    minReduceAtomic<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_result, size, identity);
    cudaDeviceSynchronize();

    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    
    // Get the final result
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return result;
}

float reductionMax(const float* d_input, int size) {
    int threadsPerBlock = BLOCK_SIZE; 
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    float identity = -INFINITY;
    
    // Allocate and initialize output value with identity
    float* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_result, &identity, sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch single kernel with all blocks
    Timer timer;
    timer.start("GPU Maximum Calculation");
    maxReduceAtomic<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_input, d_result, size, identity);
    cudaDeviceSynchronize();
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
    
    // Get the final result
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    return result;
}

int main(int argc, char* argv[]) {
    int size = 100'000; // Default value
    // Parse argument with ./reduction_test <size>
    if (argc > 1) {
        size = std::stoi(argv[1]);
    }
    std::cout << "Testing reduction with array size: " << size << std::endl;

    // Generate random data from -100.0f to 100.0f
    std::vector<float> h_data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    
    // Data Generator
    for (int i = 0; i < size; i++) {
        h_data[i] = dist(gen);
    }

    // Device mem alloc
    float* d_data;
    cudaMalloc(&d_data, size * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate CPU
    float cpu_min = 0.0f;
    Timer cpu_min_timer;
    cpu_min_timer.start("CPU Minimum Calculation");
    cpu_min = calculateMinCPU(h_data);
    cpu_min_timer.stop();
    
    float cpu_max = 0.0f;
    Timer cpu_max_timer;
    cpu_max_timer.start("CPU Maximum Calculation");
    cpu_max = calculateMaxCPU(h_data);
    cpu_max_timer.stop();
    
    // Calculate GPU
    float gpu_min = 0.0f;
    gpu_min = reductionMin(d_data, size);

    float gpu_max = 0.0f;
    gpu_max = reductionMax(d_data, size);

    // Compare results
    const float epsilon = 1e-5;
    bool min_correct = std::abs(cpu_min - gpu_min) < epsilon;
    bool max_correct = std::abs(cpu_max - gpu_max) < epsilon;

    // Output results
    std::cout << "Minimum Value:" << std::endl;
    std::cout << "  CPU: " << cpu_min << std::endl;
    std::cout << "  GPU: " << gpu_min << std::endl;
    std::cout << "  " << (min_correct ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "Maximum Value:" << std::endl;
    std::cout << "  CPU: " << cpu_max << std::endl;
    std::cout << "  GPU: " << gpu_max << std::endl;
    std::cout << "  " << (max_correct ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    
    return (min_correct && max_correct) ? 0 : 1;
}
