#include "cuda_helper.h"
#include "reduction.cuh"

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

// Perform reduction on a TriangleMesh to compute its AABB
__host__ void computeMeshReduction(const TriangleMesh* mesh_list, int mesh_idx,
                                   float* min_bounds, float* max_bounds) {
    // Access the specific mesh from mesh list
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
__host__ void computeMeshReductionStreams(const TriangleMesh* mesh_list, int mesh_idx, float* min_bounds, float* max_bounds) {
    // if (!mesh_list || mesh_list->numTriangles() <= 0) {
    //     return;
    // }

    // Access a specific mesh from the list of meshes
    const TriangleMesh* mesh = &mesh_list[mesh_idx];

    const int num_triangles = mesh->numTriangles();

    // Create device arrays for results
    float *d_min_results[9], *d_max_results[9];
    float h_min_results[9], h_max_results[9];

    // Initialize with extreme values
    float h_inf = INFINITY;
    float h_neg_inf = -INFINITY;

    for (int i = 0; i < 9; i++) {
        CUDA_CHECK(cudaMalloc(&d_min_results[i], sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_max_results[i], sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_min_results[i], &h_inf, sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_max_results[i], &h_neg_inf, sizeof(float), cudaMemcpyHostToDevice));
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

    // Point to correct vertex data with mesh->d_v0x
    // Process all vertices (v0, v1, v2) with separate streams for each dimension
    float* vertex_arrays[9] = {
        mesh->d_v0x, mesh->d_v0y, mesh->d_v0z,  // Vertex 0
        mesh->d_v1x, mesh->d_v1y, mesh->d_v1z,  // Vertex 1
        mesh->d_v2x, mesh->d_v2y, mesh->d_v2z   // Vertex 2
    };

    // Launch all min reduction kernels
    for (int i = 0; i < 9; i++) {
        minReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[i]>>>(
            vertex_arrays[i], d_min_results[i], num_triangles, INFINITY);
    }

    // Launch all max reduction kernels
    for (int i = 0; i < 9; i++) {
        maxReduceAtomic<<<grid_size, block_size, shared_mem_size, streams[i]>>>(
            vertex_arrays[i], d_max_results[i], num_triangles, -INFINITY);
    }

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
    for (int i = 0; i < 3; i++) {
        min_bounds[i] = fminf(h_min_results[i],
                              fminf(h_min_results[i + 3], h_min_results[i + 6]));  // min x
        max_bounds[i] = fmaxf(h_max_results[i],
                              fmaxf(h_max_results[i + 3], h_max_results[i + 6]));  // max x
    }
    
    // Clean up
    for (int i = 0; i < 9; i++) {
        cudaFree(d_min_results[i]);
        cudaFree(d_max_results[i]);
        cudaStreamDestroy(streams[i]);
    }
}