#ifndef AABB_CUH
#define AABB_CUH

#include "ray.cuh"

class AABB {
public:
    // Constructor using RAII
    __host__ AABB();
    __host__ AABB(int num_obj);
    __host__ ~AABB();

    __host__ void mallocAABB();
    __host__ void freeAABB();
    __host__ void cudaMallocAABB();
    __host__ void AABBMemcpyHtD();
    __host__ void AABBMemcpyDtH();
    __host__ void cudaFreeAABB();

    __device__ bool hitAABB(const Ray& __restrict__ ray, int idx) const;
    
    // Slower AABB interesction test
    __device__ bool hitAABBLoop(const Ray& ray, int idx) const;

public:
    // Making these public to simplify access from tests and applications
    float* h_minx;
    float* h_miny;
    float* h_minz;
    float* h_maxx;
    float* h_maxy;
    float* h_maxz;

    float* d_minx;
    float* d_miny;
    float* d_minz;
    float* d_maxx;
    float* d_maxy;
    float* d_maxz;

    // number of objects in the scene
    int num_obj;
};

#endif