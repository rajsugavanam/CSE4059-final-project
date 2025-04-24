#include "aabb.cuh"
#include "cuda_helper.h"
#include "reduction.cuh"
#include "timer.h"

// Atomic CAS for float
// https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ float atomicMinf(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ float atomicMaxf(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void minReduceAtomic(const float* input, float* output, int size,
                                float identity) {
    extern __shared__ float s_input[];
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    if (i + blockDim.x < size) {
        s_input[tid] = fminf(input[i], input[i + blockDim.x]);
    } else if (i < size) {
        s_input[tid] = input[i];
    } else {
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
__global__ void maxReduceAtomic(const float* input, float* output, int size,
                                float identity) {
    extern __shared__ float s_input[];
    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    if (i + blockDim.x < size) {
        s_input[tid] = fmaxf(input[i], input[i + blockDim.x]);
    } else if (i < size) {
        s_input[tid] = input[i];
    } else {
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

// Old Reduction Kernel for Triangle Mesh
__global__ void minTriangleMeshOld(const Triangle3* triangles, int num_tri,
                                   float* box_minf, float identity) {
    // Use dynamic shared memory with proper offsets
    extern __shared__ float s_min[];
    float* s_minx = s_min;
    float* s_miny = &s_min[blockDim.x];
    float* s_minz = &s_min[2 * blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    // load triangle data into shared memory

    if (i + blockDim.x < num_tri) {
        float v0x = fminf(triangles[i].vertex0().x(),
                          triangles[i + blockDim.x].vertex0().x());
        float v1x = fminf(triangles[i].vertex1().x(),
                          triangles[i + blockDim.x].vertex1().x());
        float v2x = fminf(triangles[i].vertex2().x(),
                          triangles[i + blockDim.x].vertex2().x());
        s_minx[tid] = fminf(v0x, fminf(v1x, v2x));
        float v0y = fminf(triangles[i].vertex0().y(),
                          triangles[i + blockDim.x].vertex0().y());
        float v1y = fminf(triangles[i].vertex1().y(),
                          triangles[i + blockDim.x].vertex1().y());
        float v2y = fminf(triangles[i].vertex2().y(),
                          triangles[i + blockDim.x].vertex2().y());
        s_miny[tid] = fminf(v0y, fminf(v1y, v2y));
        float v0z = fminf(triangles[i].vertex0().z(),
                          triangles[i + blockDim.x].vertex0().z());
        float v1z = fminf(triangles[i].vertex1().z(),
                          triangles[i + blockDim.x].vertex1().z());
        float v2z = fminf(triangles[i].vertex2().z(),
                          triangles[i + blockDim.x].vertex2().z());
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
    } else {
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

__global__ void maxTriangleMeshOld(const Triangle3* triangles, int num_tri,
                                   float* box_maxf, float identity) {
    // Use dynamic shared memory with proper offsets
    extern __shared__ float s_max[];
    float* s_maxx = s_max;
    float* s_maxy = &s_max[blockDim.x];
    float* s_maxz = &s_max[2 * blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    unsigned int i = segment + tid;

    // load triangle data into shared memory
    if (i + blockDim.x < num_tri) {
        float v0x = fmaxf(triangles[i].vertex0().x(),
                          triangles[i + blockDim.x].vertex0().x());
        float v1x = fmaxf(triangles[i].vertex1().x(),
                          triangles[i + blockDim.x].vertex1().x());
        float v2x = fmaxf(triangles[i].vertex2().x(),
                          triangles[i + blockDim.x].vertex2().x());
        s_maxx[tid] = fmaxf(v0x, fmaxf(v1x, v2x));
        float v0y = fmaxf(triangles[i].vertex0().y(),
                          triangles[i + blockDim.x].vertex0().y());
        float v1y = fmaxf(triangles[i].vertex1().y(),
                          triangles[i + blockDim.x].vertex1().y());
        float v2y = fmaxf(triangles[i].vertex2().y(),
                          triangles[i + blockDim.x].vertex2().y());
        s_maxy[tid] = fmaxf(v0y, fmaxf(v1y, v2y));
        float v0z = fmaxf(triangles[i].vertex0().z(),
                          triangles[i + blockDim.x].vertex0().z());
        float v1z = fmaxf(triangles[i].vertex1().z(),
                          triangles[i + blockDim.x].vertex1().z());
        float v2z = fmaxf(triangles[i].vertex2().z(),
                          triangles[i + blockDim.x].vertex2().z());
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
    } else {
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

// WARN: Havne't tested this yet JS
__global__ void reduceTriangleMesh(const float* v0x, const float* v0y,
                                   const float* v0z, const float* v1x,
                                   const float* v1y, const float* v1z,
                                   const float* v2x, const float* v2y,
                                   const float* v2z, int num_triangles,
                                   float* min_results, float* max_results) {
    extern __shared__ float s_data[];
    // Allocate shared memory for min/max coordinates
    float* s_min_x = s_data;
    float* s_min_y = s_data + blockDim.x;
    float* s_min_z = s_data + 2 * blockDim.x;
    float* s_max_x = s_data + 3 * blockDim.x;
    float* s_max_y = s_data + 4 * blockDim.x;
    float* s_max_z = s_data + 5 * blockDim.x;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // Initialize with appropriate values
    float thread_min_x = INFINITY;
    float thread_min_y = INFINITY;
    float thread_min_z = INFINITY;
    float thread_max_x = -INFINITY;
    float thread_max_y = -INFINITY;
    float thread_max_z = -INFINITY;

    // Process triangle vertices
    if (i < num_triangles) {
        // Find min and max for each triangle's vertices
        // Vertex 0
        thread_min_x = fminf(thread_min_x, v0x[i]);
        thread_min_y = fminf(thread_min_y, v0y[i]);
        thread_min_z = fminf(thread_min_z, v0z[i]);
        thread_max_x = fmaxf(thread_max_x, v0x[i]);
        thread_max_y = fmaxf(thread_max_y, v0y[i]);
        thread_max_z = fmaxf(thread_max_z, v0z[i]);

        // Vertex 1
        thread_min_x = fminf(thread_min_x, v1x[i]);
        thread_min_y = fminf(thread_min_y, v1y[i]);
        thread_min_z = fminf(thread_min_z, v1z[i]);
        thread_max_x = fmaxf(thread_max_x, v1x[i]);
        thread_max_y = fmaxf(thread_max_y, v1y[i]);
        thread_max_z = fmaxf(thread_max_z, v1z[i]);

        // Vertex 2
        thread_min_x = fminf(thread_min_x, v2x[i]);
        thread_min_y = fminf(thread_min_y, v2y[i]);
        thread_min_z = fminf(thread_min_z, v2z[i]);
        thread_max_x = fmaxf(thread_max_x, v2x[i]);
        thread_max_y = fmaxf(thread_max_y, v2y[i]);
        thread_max_z = fmaxf(thread_max_z, v2z[i]);
    }

    // Store in shared memory
    s_min_x[tid] = thread_min_x;
    s_min_y[tid] = thread_min_y;
    s_min_z[tid] = thread_min_z;
    s_max_x[tid] = thread_max_x;
    s_max_y[tid] = thread_max_y;
    s_max_z[tid] = thread_max_z;

    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min_x[tid] = fminf(s_min_x[tid], s_min_x[tid + s]);
            s_min_y[tid] = fminf(s_min_y[tid], s_min_y[tid + s]);
            s_min_z[tid] = fminf(s_min_z[tid], s_min_z[tid + s]);
            s_max_x[tid] = fmaxf(s_max_x[tid], s_max_x[tid + s]);
            s_max_y[tid] = fmaxf(s_max_y[tid], s_max_y[tid + s]);
            s_max_z[tid] = fmaxf(s_max_z[tid], s_max_z[tid + s]);
        }
        __syncthreads();
    }

    // Write results to global memory
    if (tid == 0) {
        atomicMinf(&min_results[0], s_min_x[0]);
        atomicMinf(&min_results[1], s_min_y[0]);
        atomicMinf(&min_results[2], s_min_z[0]);
        atomicMaxf(&max_results[0], s_max_x[0]);
        atomicMaxf(&max_results[1], s_max_y[0]);
        atomicMaxf(&max_results[2], s_max_z[0]);
    }
}

// Modified computeMeshReduction function to directly use triangle data
__host__ void computeMeshReduction(const TriangleMesh* mesh, float* min_bounds,
                                   float* max_bounds) {
    if (!mesh || mesh->numTriangles() <= 0) {
        // Invalid mesh
        return;
    }

    int num_triangles = mesh->numTriangles();

    // Get raw vertex pointers from the mesh
    const float *h_v0x, *h_v0y, *h_v0z;
    const float *h_v1x, *h_v1y, *h_v1z;
    const float *h_v2x, *h_v2y, *h_v2z;

    mesh->getRawVertexPointers(&h_v0x, &h_v0y, &h_v0z, &h_v1x, &h_v1y, &h_v1z,
                               &h_v2x, &h_v2y, &h_v2z);

    // Allocate device memory for vertex data
    float *d_v0x, *d_v0y, *d_v0z;
    float *d_v1x, *d_v1y, *d_v1z;
    float *d_v2x, *d_v2y, *d_v2z;
    float *d_min_results, *d_max_results;

    cudaMalloc(&d_v0x, num_triangles * sizeof(float));
    cudaMalloc(&d_v0y, num_triangles * sizeof(float));
    cudaMalloc(&d_v0z, num_triangles * sizeof(float));
    cudaMalloc(&d_v1x, num_triangles * sizeof(float));
    cudaMalloc(&d_v1y, num_triangles * sizeof(float));
    cudaMalloc(&d_v1z, num_triangles * sizeof(float));
    cudaMalloc(&d_v2x, num_triangles * sizeof(float));
    cudaMalloc(&d_v2y, num_triangles * sizeof(float));
    cudaMalloc(&d_v2z, num_triangles * sizeof(float));
    cudaMalloc(&d_min_results, 3 * sizeof(float));
    cudaMalloc(&d_max_results, 3 * sizeof(float));

    // Copy vertex data to device
    cudaMemcpy(d_v0x, h_v0x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0y, h_v0y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0z, h_v0z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1x, h_v1x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1y, h_v1y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1z, h_v1z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2x, h_v2x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2y, h_v2y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2z, h_v2z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);

    // Initialize min/max results
    float h_inf = INFINITY;
    float h_neg_inf = -INFINITY;
    float h_min_init[3] = {h_inf, h_inf, h_inf};
    float h_max_init[3] = {h_neg_inf, h_neg_inf, h_neg_inf};
    cudaMemcpy(d_min_results, h_min_init, 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_results, h_max_init, 3 * sizeof(float),
               cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 block_size(256);
    dim3 grid_size((num_triangles + block_size.x - 1) / block_size.x);
    size_t shared_mem_size =
        6 * block_size.x * sizeof(float);  // 6 arrays: min/max for x,y,z

    // Launch the kernel
    reduceTriangleMesh<<<grid_size, block_size, shared_mem_size>>>(
        d_v0x, d_v0y, d_v0z, d_v1x, d_v1y, d_v1z, d_v2x, d_v2y, d_v2z,
        num_triangles, d_min_results, d_max_results);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(min_bounds, d_min_results, 3 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(max_bounds, d_max_results, 3 * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_v0x);
    cudaFree(d_v0y);
    cudaFree(d_v0z);
    cudaFree(d_v1x);
    cudaFree(d_v1y);
    cudaFree(d_v1z);
    cudaFree(d_v2x);
    cudaFree(d_v2y);
    cudaFree(d_v2z);
    cudaFree(d_min_results);
    cudaFree(d_max_results);
}

// Perform reduction on a TriangleMesh to compute its AABB
__host__ void computeMeshReduction(const TriangleMesh* mesh_list, int mesh_idx,
                                   float* min_bounds, float* max_bounds) {
    // Access the specific mesh from the collection
    const TriangleMesh* mesh = &mesh_list[mesh_idx];

    if (!mesh || mesh->numTriangles() <= 0) {
        // Invalid mesh
        return;
    }

    // Get pointers to the vertex data
    const float *h_v0x, *h_v0y, *h_v0z;
    const float *h_v1x, *h_v1y, *h_v1z;
    const float *h_v2x, *h_v2y, *h_v2z;

    mesh->getRawVertexPointers(&h_v0x, &h_v0y, &h_v0z, &h_v1x, &h_v1y, &h_v1z,
                               &h_v2x, &h_v2y, &h_v2z);

    // Create arrays to hold all x, y, z coordinates for reduction
    int num_triangles = mesh->numTriangles();
    int num_vertices = num_triangles * 3;
    float* all_x = new float[num_vertices];
    float* all_y = new float[num_vertices];
    float* all_z = new float[num_vertices];

    // Fill the arrays with all vertex coordinates
    for (int i = 0; i < num_triangles; i++) {
        all_x[i * 3] = h_v0x[i];
        all_x[i * 3 + 1] = h_v1x[i];
        all_x[i * 3 + 2] = h_v2x[i];

        all_y[i * 3] = h_v0y[i];
        all_y[i * 3 + 1] = h_v1y[i];
        all_y[i * 3 + 2] = h_v2y[i];

        all_z[i * 3] = h_v0z[i];
        all_z[i * 3 + 1] = h_v1z[i];
        all_z[i * 3 + 2] = h_v2z[i];
    }

    // Allocate memory on the device
    float *d_all_x, *d_all_y, *d_all_z;
    float *d_min_x, *d_min_y, *d_min_z;
    float *d_max_x, *d_max_y, *d_max_z;

    cudaMalloc(&d_all_x, num_vertices * sizeof(float));
    cudaMalloc(&d_all_y, num_vertices * sizeof(float));
    cudaMalloc(&d_all_z, num_vertices * sizeof(float));
    cudaMalloc(&d_min_x, sizeof(float));
    cudaMalloc(&d_min_y, sizeof(float));
    cudaMalloc(&d_min_z, sizeof(float));
    cudaMalloc(&d_max_x, sizeof(float));
    cudaMalloc(&d_max_y, sizeof(float));
    cudaMalloc(&d_max_z, sizeof(float));

    // Initialize min/max values
    float h_inf = INFINITY;
    float h_neg_inf = -INFINITY;
    cudaMemcpy(d_min_x, &h_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_y, &h_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_z, &h_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_x, &h_neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_y, &h_neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_z, &h_neg_inf, sizeof(float), cudaMemcpyHostToDevice);

    // Copy vertex data to device
    cudaMemcpy(d_all_x, all_x, num_vertices * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_y, all_y, num_vertices * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_z, all_z, num_vertices * sizeof(float),
               cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 block_size(256);
    dim3 grid_size((num_vertices + block_size.x - 1) / block_size.x);
    size_t shared_mem_size = block_size.x * sizeof(float);

    // Launch reduction kernels
    minReduceAtomic<<<grid_size, block_size, shared_mem_size>>>(
        d_all_x, d_min_x, num_vertices, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size>>>(
        d_all_y, d_min_y, num_vertices, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size>>>(
        d_all_z, d_min_z, num_vertices, INFINITY);

    maxReduceAtomic<<<grid_size, block_size, shared_mem_size>>>(
        d_all_x, d_max_x, num_vertices, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size>>>(
        d_all_y, d_max_y, num_vertices, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size>>>(
        d_all_z, d_max_z, num_vertices, -INFINITY);

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(&min_bounds[0], d_min_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_bounds[1], d_min_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_bounds[2], d_min_z, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_bounds[0], d_max_x, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_bounds[1], d_max_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_bounds[2], d_max_z, sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] all_x;
    delete[] all_y;
    delete[] all_z;
    cudaFree(d_all_x);
    cudaFree(d_all_y);
    cudaFree(d_all_z);
    cudaFree(d_min_x);
    cudaFree(d_min_y);
    cudaFree(d_min_z);
    cudaFree(d_max_x);
    cudaFree(d_max_y);
    cudaFree(d_max_z);
}

// Perform reduction on TriangleMesh using multiple CUDA streams for parallelism
__host__ void computeMeshReductionStreams(const TriangleMesh* mesh_list,
                                          int mesh_idx, float* min_bounds,
                                          float* max_bounds) {
    // Access the specific mesh from the collection
    const TriangleMesh* mesh = &mesh_list[mesh_idx];

    if (!mesh || mesh->numTriangles() <= 0) {
        return;
    }

    int num_triangles = mesh->numTriangles();

    // Get device pointers (if already on GPU)
    float *d_v0x, *d_v0y, *d_v0z;
    float *d_v1x, *d_v1y, *d_v1z;
    float *d_v2x, *d_v2y, *d_v2z;

    mesh->getDeviceVertexPointers(&d_v0x, &d_v0y, &d_v0z, &d_v1x, &d_v1y,
                                  &d_v1z, &d_v2x, &d_v2y, &d_v2z);

    // Create device arrays for results (9 arrays for min/max of x,y,z for each
    // vertex)
    float *d_min_results[9], *d_max_results[9];
    float h_min_results[9], h_max_results[9];

    // Initialize with extreme values
    float h_inf = INFINITY;
    float h_neg_inf = -INFINITY;

    for (int i = 0; i < 9; i++) {
        CUDA_CHECK(cudaMalloc(&d_min_results[i], sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_max_results[i], sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_min_results[i], &h_inf, sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_max_results[i], &h_neg_inf, sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // Create CUDA streams for parallel execution
    cudaStream_t streams[9];
    for (int i = 0; i < 9; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // Configure kernel launch parameters
    dim3 block_size(256);
    dim3 grid_size((num_triangles + block_size.x - 1) / block_size.x);
    size_t shared_mem_size = block_size.x * sizeof(float);

    // Launch kernels in parallel streams
    // Vertex 0 (streams 0-2)
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[0]>>>(
        d_v0x, d_min_results[0], num_triangles, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[1]>>>(
        d_v0y, d_min_results[1], num_triangles, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[2]>>>(
        d_v0z, d_min_results[2], num_triangles, INFINITY);

    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[0]>>>(
        d_v0x, d_max_results[0], num_triangles, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[1]>>>(
        d_v0y, d_max_results[1], num_triangles, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[2]>>>(
        d_v0z, d_max_results[2], num_triangles, -INFINITY);

    // Vertex 1 (streams 3-5)
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[3]>>>(
        d_v1x, d_min_results[3], num_triangles, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[4]>>>(
        d_v1y, d_min_results[4], num_triangles, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[5]>>>(
        d_v1z, d_min_results[5], num_triangles, INFINITY);

    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[3]>>>(
        d_v1x, d_max_results[3], num_triangles, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[4]>>>(
        d_v1y, d_max_results[4], num_triangles, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[5]>>>(
        d_v1z, d_max_results[5], num_triangles, -INFINITY);

    // Vertex 2 (streams 6-8)
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[6]>>>(
        d_v2x, d_min_results[6], num_triangles, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[7]>>>(
        d_v2y, d_min_results[7], num_triangles, INFINITY);
    minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[8]>>>(
        d_v2z, d_min_results[8], num_triangles, INFINITY);

    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[6]>>>(
        d_v2x, d_max_results[6], num_triangles, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[7]>>>(
        d_v2y, d_max_results[7], num_triangles, -INFINITY);
    maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[8]>>>(
        d_v2z, d_max_results[8], num_triangles, -INFINITY);

    // Synchronize all streams
    for (int i = 0; i < 9; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // Copy results back to host
    for (int i = 0; i < 9; i++) {
        CUDA_CHECK(cudaMemcpy(&h_min_results[i], d_min_results[i],
                              sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_max_results[i], d_max_results[i],
                              sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Find global min/max across all vertices
    min_bounds[0] = fminf(h_min_results[0],
                          fminf(h_min_results[3], h_min_results[6]));  // min x
    min_bounds[1] = fminf(h_min_results[1],
                          fminf(h_min_results[4], h_min_results[7]));  // min y
    min_bounds[2] = fminf(h_min_results[2],
                          fminf(h_min_results[5], h_min_results[8]));  // min z

    max_bounds[0] = fmaxf(h_max_results[0],
                          fmaxf(h_max_results[3], h_max_results[6]));  // max x
    max_bounds[1] = fmaxf(h_max_results[1],
                          fmaxf(h_max_results[4], h_max_results[7]));  // max y
    max_bounds[2] = fmaxf(h_max_results[2],
                          fmaxf(h_max_results[5], h_max_results[8]));  // max z

    // Clean up
    for (int i = 0; i < 9; i++) {
        cudaFree(d_min_results[i]);
        cudaFree(d_max_results[i]);
        cudaStreamDestroy(streams[i]);
    }
}