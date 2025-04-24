#ifndef REDUCTION_CUH
#define REDUCTION_CUH

#include <cuda_runtime.h>
#include <triangle3.cuh>
#include <triangle_mesh.cuh>

// Kernel function declarations
__device__ float atomicMinf(float* address, float val);
__device__ float atomicMaxf(float* address, float val);
__global__ void minReduceAtomic(const float* input, float* output, int size, float identity);
__global__ void maxReduceAtomic(const float* input, float* output, int size, float identity);

// __global__ void minTriangleMesh(const TriangleMesh* tri_mesh, int num_tri,
//                                 float* box_minf, float identity);
// __global__ void maxTriangleMesh(const TriangleMesh* tri_mesh, int num_tri,
//                                 float* box_minf, float identity);

__global__ void minTriangleMeshOld(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity);
__global__ void maxTriangleMeshOld(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity);

// Add the TriangleMesh reduction function declaration
__host__ void computeMeshReduction(const TriangleMesh* mesh_list, int mesh_idx, float* min_bounds, float* max_bounds);

// Add the new stream-based reduction function declaration
__host__ void computeMeshReductionStreams(const TriangleMesh* mesh_list, int mesh_idx, float* min_bounds, float* max_bounds);


#endif // REDUCTION_CUH