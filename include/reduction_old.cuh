#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <triangle3.cuh>

// Kernel function declarations
__device__ float atomicMinf(float* address, float val);
__device__ float atomicMaxf(float* address, float val);
__global__ void minReduceAtomic(const float* input, float* output, int size, float identity);
__global__ void maxReduceAtomic(const float* input, float* output, int size, float identity);

__global__ void minTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity);

__global__ void maxTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity);
#endif // REDUCTION_CUH