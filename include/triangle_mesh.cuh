#ifndef TRIANGLE_MESH_SOA_CUH
#define TRIANGLE_MESH_SOA_CUH

#include <cuda_runtime.h>

#include "triangle3.cuh"
#include "vec3.cuh"

class TriangleMeshSoA {
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

    // Size information
    int size;

    // Constructors
    __host__ TriangleMeshSoA()
        : size(0),
          h_v0x(nullptr),
          h_v0y(nullptr),
          h_v0z(nullptr),
          h_v1x(nullptr),
          h_v1y(nullptr),
          h_v1z(nullptr),
          h_v2x(nullptr),
          h_v2y(nullptr),
          h_v2z(nullptr),
          h_n0x(nullptr),
          h_n0y(nullptr),
          h_n0z(nullptr),
          h_n1x(nullptr),
          h_n1y(nullptr),
          h_n1z(nullptr),
          h_n2x(nullptr),
          h_n2y(nullptr),
          h_n2z(nullptr),
          d_v0x(nullptr),
          d_v0y(nullptr),
          d_v0z(nullptr),
          d_v1x(nullptr),
          d_v1y(nullptr),
          d_v1z(nullptr),
          d_v2x(nullptr),
          d_v2y(nullptr),
          d_v2z(nullptr),
          d_n0x(nullptr),
          d_n0y(nullptr),
          d_n0z(nullptr),
          d_n1x(nullptr),
          d_n1y(nullptr),
          d_n1z(nullptr),
          d_n2x(nullptr),
          d_n2y(nullptr),
          d_n2z(nullptr) {}

    __host__ TriangleMeshSoA(int triangleCount) : size(triangleCount) {
        allocateHostMemory();
        allocateDeviceMemory();
    }

    // Destructor
    __host__ ~TriangleMeshSoA() {
        freeHostMemory(); 
        freeDeviceMemory();
    }

    // Allocate host memory
    __host__ void allocateHostMemory() {
        if (size > 0) {
            // Vertices
            h_v0x = new float[size];
            h_v0y = new float[size];
            h_v0z = new float[size];
            h_v1x = new float[size];
            h_v1y = new float[size];
            h_v1z = new float[size];
            h_v2x = new float[size];
            h_v2y = new float[size];
            h_v2z = new float[size];

            // Normals
            h_n0x = new float[size];
            h_n0y = new float[size];
            h_n0z = new float[size];
            h_n1x = new float[size];
            h_n1y = new float[size];
            h_n1z = new float[size];
            h_n2x = new float[size];
            h_n2y = new float[size];
            h_n2z = new float[size];
        }
    }

    // Allocate device memory
    __host__ void allocateDeviceMemory() {
        if (size > 0) {
            // Vertices
            cudaMalloc(&d_v0x, size * sizeof(float));
            cudaMalloc(&d_v0y, size * sizeof(float));
            cudaMalloc(&d_v0z, size * sizeof(float));
            cudaMalloc(&d_v1x, size * sizeof(float));
            cudaMalloc(&d_v1y, size * sizeof(float));
            cudaMalloc(&d_v1z, size * sizeof(float));
            cudaMalloc(&d_v2x, size * sizeof(float));
            cudaMalloc(&d_v2y, size * sizeof(float));
            cudaMalloc(&d_v2z, size * sizeof(float));

            // Normals
            cudaMalloc(&d_n0x, size * sizeof(float));
            cudaMalloc(&d_n0y, size * sizeof(float));
            cudaMalloc(&d_n0z, size * sizeof(float));
            cudaMalloc(&d_n1x, size * sizeof(float));
            cudaMalloc(&d_n1y, size * sizeof(float));
            cudaMalloc(&d_n1z, size * sizeof(float));
            cudaMalloc(&d_n2x, size * sizeof(float));
            cudaMalloc(&d_n2y, size * sizeof(float));
            cudaMalloc(&d_n2z, size * sizeof(float));
        }
    }

    // Free host memory
    __host__ void freeHostMemory() {
        // Vertices
        delete[] h_v0x;
        delete[] h_v0y;
        delete[] h_v0z;
        delete[] h_v1x;
        delete[] h_v1y;
        delete[] h_v1z;
        delete[] h_v2x;
        delete[] h_v2y;
        delete[] h_v2z;

        // Normals
        delete[] h_n0x;
        delete[] h_n0y;
        delete[] h_n0z;
        delete[] h_n1x;
        delete[] h_n1y;
        delete[] h_n1z;
        delete[] h_n2x;
        delete[] h_n2y;
        delete[] h_n2z;

        h_v0x = h_v0y = h_v0z = nullptr;
        h_v1x = h_v1y = h_v1z = nullptr;
        h_v2x = h_v2y = h_v2z = nullptr;
        h_n0x = h_n0y = h_n0z = nullptr;
        h_n1x = h_n1y = h_n1z = nullptr;
        h_n2x = h_n2y = h_n2z = nullptr;
    }

    // Convert from Array of Structures to Structure of Arrays for now...
    __host__ static TriangleMeshSoA fromTriangleArray(
        const Triangle3* triangles, int count) {
        TriangleMeshSoA mesh(count);

        for (int i = 0; i < count; i++) {
            // Copy vertices
            mesh.h_v0x[i] = triangles[i].vertex0().x();
            mesh.h_v0y[i] = triangles[i].vertex0().y();
            mesh.h_v0z[i] = triangles[i].vertex0().z();

            mesh.h_v1x[i] = triangles[i].vertex1().x();
            mesh.h_v1y[i] = triangles[i].vertex1().y();
            mesh.h_v1z[i] = triangles[i].vertex1().z();

            mesh.h_v2x[i] = triangles[i].vertex2().x();
            mesh.h_v2y[i] = triangles[i].vertex2().y();
            mesh.h_v2z[i] = triangles[i].vertex2().z();

            // Copy normals
            mesh.h_n0x[i] = triangles[i].normal0().x();
            mesh.h_n0y[i] = triangles[i].normal0().y();
            mesh.h_n0z[i] = triangles[i].normal0().z();

            mesh.h_n1x[i] = triangles[i].normal1().x();
            mesh.h_n1y[i] = triangles[i].normal1().y();
            mesh.h_n1z[i] = triangles[i].normal1().z();

            mesh.h_n2x[i] = triangles[i].normal2().x();
            mesh.h_n2y[i] = triangles[i].normal2().y();
            mesh.h_n2z[i] = triangles[i].normal2().z();
        }

        return mesh;
    }

    // CUDA device memory management
    __host__ void meshMemcpy() {

        // Copy data to device
        cudaMemcpy(d_v0x, h_v0x, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v0y, h_v0y, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v0z, h_v0z, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1x, h_v1x, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1y, h_v1y, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1z, h_v1z, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2x, h_v2x, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2y, h_v2y, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2z, h_v2z, size * sizeof(float), cudaMemcpyHostToDevice);

        // Copy normals
        cudaMemcpy(d_n0x, h_n0x, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n0y, h_n0y, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n0z, h_n0z, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n1x, h_n1x, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n1y, h_n1y, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n1z, h_n1z, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n2x, h_n2x, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n2y, h_n2y, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n2z, h_n2z, size * sizeof(float), cudaMemcpyHostToDevice);

    }

    __host__ void freeDeviceMemory() {
        cudaFree(d_v0x);
        cudaFree(d_v0y);
        cudaFree(d_v0z);
        cudaFree(d_v1x);
        cudaFree(d_v1y);
        cudaFree(d_v1z);
        cudaFree(d_v2x);
        cudaFree(d_v2y);
        cudaFree(d_v2z);
        cudaFree(d_n0x);
        cudaFree(d_n0y);
        cudaFree(d_n0z);
        cudaFree(d_n1x);
        cudaFree(d_n1y);
        cudaFree(d_n1z);
        cudaFree(d_n2x);
        cudaFree(d_n2y);
        cudaFree(d_n2z);
    }
};

#endif  // TRIANGLE_MESH_SOA_CUH