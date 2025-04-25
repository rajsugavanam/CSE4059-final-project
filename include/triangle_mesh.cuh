#ifndef TRIANGLE_MESH_CUH
#define TRIANGLE_MESH_CUH

#include <cuda_runtime.h>

#include "aabb.cuh"

class TriangleMesh {
   public:
    // Constructors
    __host__ TriangleMesh();
    __host__ TriangleMesh(int num_triangles);

    // Destructor
    __host__ ~TriangleMesh();

    // Host memory management
    __host__ void mallocTriangleMesh();
    __host__ void freeTriangleMesh();
    __host__ void cudaMallocTriangleMesh();

    // CUDA device memory management
    __host__ void meshMemcpyHtD();
    __host__ void cudaFreeTriangleMesh();

    // New method to load from OBJ file
    __host__ static TriangleMesh* loadFromOBJ(const std::string& filename);
    
    __host__ __device__ int numTriangles() const { return num_triangles; }

    // Compute the AABB and store it in the provided pointer
    __host__ void computeAABB(AABB* aabb, int obj_id);

    // Get raw pointers to vertex data
    __host__ void getRawVertexPointers(
        const float** v0x, const float** v0y, const float** v0z,
        const float** v1x, const float** v1y, const float** v1z,
        const float** v2x, const float** v2y, const float** v2z) const;
    
    // Get pointers to device vertex data
    __host__ void getDeviceVertexPointers(
        float** v0x, float** v0y, float** v0z,
        float** v1x, float** v1y, float** v1z,
        float** v2x, float** v2y, float** v2z) const;

   public:
   // Host pointers
    // Vertex positions
    float* h_v0x;
    float* h_v0y;
    float* h_v0z;
    float* h_v1x;
    float* h_v1y;
    float* h_v1z;
    float* h_v2x;
    float* h_v2y;
    float* h_v2z;

    // Normal vectors
    float* h_n0x;
    float* h_n0y;
    float* h_n0z;
    float* h_n1x;
    float* h_n1y;
    float* h_n1z;
    float* h_n2x;
    float* h_n2y;
    float* h_n2z;

    // Device pointers
    float* d_v0x;
    float* d_v0y;
    float* d_v0z;
    float* d_v1x;
    float* d_v1y;
    float* d_v1z;
    float* d_v2x;
    float* d_v2y;
    float* d_v2z;

    float* d_n0x;
    float* d_n0y;
    float* d_n0z;
    float* d_n1x;
    float* d_n1y;
    float* d_n1z;
    float* d_n2x;
    float* d_n2y;
    float* d_n2z;

    // Number of triangles in the mesh
    int num_triangles;
};

#endif  // TRIANGLE_MESH_CUH
