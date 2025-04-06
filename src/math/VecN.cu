#include "VecN.cuh"
#include <cassert>
#include <stdexcept>


template<typename T>
VecN<T>::~VecN<T>() {
    delete[] pv;
}

template<typename T>
__global__ void kerAdd(T* oper1, T* oper2, T* res, int N) {
    // most boring thread mapping ever.
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid >= N) {
        return;
    }
    res[tid] = oper2[tid] + oper1[tid];
}

template<typename T>
VecN<T>* VecN<T>::deviceAdd(VecN<T>* vec2) {

    cudaError_t lastError = cudaGetLastError();

    if (this->N != vec2->N) {
        throw std::runtime_error(
                "[ERROR, VecN::deviceAdd()]:  Cannot add vectors of mismatched length!"
                );
    }


    // allocate memory for device oper1, oper2, res.

    T* devOper1;
    T* devOper2;
    T* devRes;

    int vecBytes = N*sizeof(T);
    cudaMalloc(&devOper1, vecBytes);
    cudaMalloc(&devOper2, vecBytes);
    cudaMalloc(&devRes, vecBytes);

    lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        throw std::runtime_error("[ERROR, VecN::deviceAdd()]: Failed to allocate memory!");
    }

    // transfer this to device, transfer vec2 to device.

    cudaMemcpy(devOper1, this->pv, vecBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devOper2, vec2->pv, vecBytes, cudaMemcpyHostToDevice);

    lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        throw std::runtime_error("[ERROR, VecN::deviceAdd()]: Failed to transfer host memory to device!");
    }

    // launch kernel.

    dim3 blockDim = dim3( BLOCK_SIZE, 1, 1 );
    dim3 gridDim = dim3( (N+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1 );

    kerAdd<<<gridDim, blockDim>>>(devOper1, devOper2, devRes, N);

    // transfer devRes to host.
    T* pv = (T*) malloc(vecBytes);
    cudaMemcpy(pv, devRes, vecBytes, cudaMemcpyDeviceToHost);

    VecN<T>* res = new VecN<T>(N, pv);

    // free all GPU memory.
    cudaFree(devOper1);
    cudaFree(devOper2);
    cudaFree(devRes);
    lastError = cudaGetLastError();
    if (lastError != cudaSuccess) {
        throw std::runtime_error("[ERROR, VecN::deviceAdd()]: Failed to free device memory!");
    }

    return res;
}

template class VecN<float>;
template class VecN<int>;
template __global__ void kerAdd(float*, float*, float*, int);
