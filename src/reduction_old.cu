#include "reduction.cuh"
#include "cuda_helper.h"
#include "aabb.cuh"
#include "timer.h"

// Atomic CAS for float
// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ float atomicMinf(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ float atomicMaxf(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void minReduceAtomic(const float* input, float* output, int size, float identity) {
    extern __shared__ float s_input[];
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    if (i + blockDim.x < size) {
        s_input[tid] = fminf(input[i], input[i + blockDim.x]);
    } else if (i < size) {
        s_input[tid] = input[i];
    } 
    else {
        s_input[tid] = identity;
    }

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();

        if (tid < s) {
            s_input[tid] = fminf(s_input[tid], s_input[tid + s]);
        }
    }
    
    // Multi-block reduction
    if (tid == 0) {
        atomicMinf(output, s_input[0]);
    }
}

// Modified maxReduceAtomic kernel with boundary checks and error handling
__global__ void maxReduceAtomic(const float* input, float* output, int size, float identity) {
    extern __shared__ float s_input[];
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    if (i + blockDim.x < size) {
        s_input[tid] = fmaxf(input[i], input[i + blockDim.x]);
    } else if (i < size) {
        s_input[tid] = input[i];
    } 
    else {
        s_input[tid] = identity;
    }
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();

        if (tid < s) {
            s_input[tid] = fmaxf(s_input[tid], s_input[tid + s]);
        }
    }
    
    // Multi-block reduction
    if (tid == 0) {
        atomicMaxf(output, s_input[0]);
    }
}


__global__ void minTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity) {
    // Use dynamic shared memory with proper offsets
    extern __shared__ float shared_mem[];
    float* s_minx = shared_mem;
    float* s_miny = &shared_mem[blockDim.x];
    float* s_minz = &shared_mem[2 * blockDim.x];
    
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    // load triangle data into shared memory

    if (i + blockDim.x < num_tri) {
        float v0x = fminf(triangles[i].vertex0().x(), triangles[i + blockDim.x].vertex0().x());
        float v1x = fminf(triangles[i].vertex1().x(), triangles[i + blockDim.x].vertex1().x());
        float v2x = fminf(triangles[i].vertex2().x(), triangles[i + blockDim.x].vertex2().x());
        s_minx[tid] = fminf(v0x, fminf(v1x, v2x));
        float v0y = fminf(triangles[i].vertex0().y(), triangles[i + blockDim.x].vertex0().y());
        float v1y = fminf(triangles[i].vertex1().y(), triangles[i + blockDim.x].vertex1().y());
        float v2y = fminf(triangles[i].vertex2().y(), triangles[i + blockDim.x].vertex2().y());
        s_miny[tid] = fminf(v0y, fminf(v1y, v2y));
        float v0z = fminf(triangles[i].vertex0().z(), triangles[i + blockDim.x].vertex0().z());
        float v1z = fminf(triangles[i].vertex1().z(), triangles[i + blockDim.x].vertex1().z());
        float v2z = fminf(triangles[i].vertex2().z(), triangles[i + blockDim.x].vertex2().z());
        s_minz[tid] = fminf(v0z, fminf(v1z, v2z));
    } else if (i < num_tri) {
        float v0x = triangles[i].vertex0().x();
        float v1x = triangles[i].vertex1().x();
        float v2x = triangles[i].vertex2().x();
        s_minx[tid] = fminf(v0x, fminf(v1x, v2x));
        float v0y = triangles[i].vertex0().y();
        float v1y = triangles[i].vertex1().y();
        float v2y = triangles[i].vertex2().y();
        s_miny[tid] = fminf(v0y, fminf(v1y, v2y));
        float v0z = triangles[i].vertex0().z();
        float v1z = triangles[i].vertex1().z();
        float v2z = triangles[i].vertex2().z();
        s_minz[tid] = fminf(v0z, fminf(v1z, v2z));
    } 
    else {
        s_minx[tid] = identity;
        s_miny[tid] = identity;
        s_minz[tid] = identity;
    }

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();

        if (tid < s) {
            s_minx[tid] = fminf(s_minx[tid], s_minx[tid + s]);
            s_miny[tid] = fminf(s_miny[tid], s_miny[tid + s]);
            s_minz[tid] = fminf(s_minz[tid], s_minz[tid + s]);
        }
    }
    // Multi-block reduction
    if (tid == 0) {
        atomicMinf(&box_minf[0], s_minx[0]);
        atomicMinf(&box_minf[1], s_miny[0]);
        atomicMinf(&box_minf[2], s_minz[0]);
    }
}

__global__ void maxTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_maxf, float identity) {
    // Use dynamic shared memory with proper offsets
    extern __shared__ float shared_mem[];
    float* s_maxx = shared_mem;
    float* s_maxy = &shared_mem[blockDim.x];
    float* s_maxz = &shared_mem[2 * blockDim.x];
    
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    // load triangle data into shared memory
    if (i + blockDim.x < num_tri) {
        float v0x = fmaxf(triangles[i].vertex0().x(), triangles[i + blockDim.x].vertex0().x());
        float v1x = fmaxf(triangles[i].vertex1().x(), triangles[i + blockDim.x].vertex1().x());
        float v2x = fmaxf(triangles[i].vertex2().x(), triangles[i + blockDim.x].vertex2().x());
        s_maxx[tid] = fmaxf(v0x, fmaxf(v1x, v2x));
        float v0y = fmaxf(triangles[i].vertex0().y(), triangles[i + blockDim.x].vertex0().y());
        float v1y = fmaxf(triangles[i].vertex1().y(), triangles[i + blockDim.x].vertex1().y());
        float v2y = fmaxf(triangles[i].vertex2().y(), triangles[i + blockDim.x].vertex2().y());
        s_maxy[tid] = fmaxf(v0y, fmaxf(v1y, v2y));
        float v0z = fmaxf(triangles[i].vertex0().z(), triangles[i + blockDim.x].vertex0().z());
        float v1z = fmaxf(triangles[i].vertex1().z(), triangles[i + blockDim.x].vertex1().z());
        float v2z = fmaxf(triangles[i].vertex2().z(), triangles[i + blockDim.x].vertex2().z());
        s_maxz[tid] = fmaxf(v0z, fmaxf(v1z, v2z));
    } else if (i < num_tri) {
        float v0x = triangles[i].vertex0().x();
        float v1x = triangles[i].vertex1().x();
        float v2x = triangles[i].vertex2().x();
        s_maxx[tid] = fmaxf(v0x, fmaxf(v1x, v2x));
        float v0y = triangles[i].vertex0().y();
        float v1y = triangles[i].vertex1().y();
        float v2y = triangles[i].vertex2().y();
        s_maxy[tid] = fmaxf(v0y, fmaxf(v1y, v2y));
        float v0z = triangles[i].vertex0().z();
        float v1z = triangles[i].vertex1().z();
        float v2z = triangles[i].vertex2().z();
        s_maxz[tid] = fmaxf(v0z, fmaxf(v1z, v2z));
    }
    else {
        s_maxx[tid] = identity;
        s_maxy[tid] = identity;
        s_maxz[tid] = identity;
    }
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();

        if (tid < s) {
            s_maxx[tid] = fmaxf(s_maxx[tid], s_maxx[tid + s]);
            s_maxy[tid] = fmaxf(s_maxy[tid], s_maxy[tid + s]);
            s_maxz[tid] = fmaxf(s_maxz[tid], s_maxz[tid + s]);
        }
    }
    // Multi-block reduction
    if (tid == 0) {
        atomicMaxf(&box_maxf[0], s_maxx[0]);
        atomicMaxf(&box_maxf[1], s_maxy[0]);
        atomicMaxf(&box_maxf[2], s_maxz[0]);
    }
}