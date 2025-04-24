#include "light_sampling.cuh"

#define CURAND_SEED 12312

__global__ void kerInitRng(curandState* curand_state, int pix_height, int pix_width) {
    unsigned int tx = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int ty = blockDim.y*blockIdx.y + threadIdx.y;
    if (tx >= pix_width || ty >= pix_height) {
        return;
    }
    unsigned int t = pix_width*ty + tx;
    curand_init(CURAND_SEED, t, 0, &curand_state[t]);
}

__global__ void kerCosineRng(float* values, curandState* curand_state, int pix_height, int pix_width) {
    unsigned int tx = blockDim.x*blockIdx.x + threadIdx.x;
    unsigned int ty = blockDim.y*blockIdx.y + threadIdx.y;
    if (tx >= pix_width || ty >= pix_height) {
        return;
    }
    unsigned int t = pix_width*ty + tx;
    float uniform_value = curand_uniform(&curand_state[t]);
    float transformed = asin(2*uniform_value-1);
    values[t] = transformed;
}

// Sampler::Sampler(const Vec3& input_ray, const Triangle3* hit_tri,
//                  const Vec3& bary_hit_point)
//     : input_ray{input_ray}, hit_tri{hit_tri}, bary_hit_point{bary_hit_point} {}

Sampler::Sampler(int kernel_block_size, int pix_width, int pix_height) : KERNEL_BLOCK_SIZE(kernel_block_size), pix_width(pix_width), pix_height(pix_height) {}

Sampler::~Sampler() {
    cudaFree(d_curandstate);
}

curandState* Sampler::initRng() {
    cudaMalloc(&d_curandstate, sizeof(curandState)*pix_height*pix_width);

    dim3 blockDim(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE, 1);
    dim3 gridDim( (pix_width+KERNEL_BLOCK_SIZE-1)/KERNEL_BLOCK_SIZE, (pix_height+KERNEL_BLOCK_SIZE-1)/KERNEL_BLOCK_SIZE, 1 );
    kerInitRng<<<gridDim, blockDim>>>(d_curandstate, pix_height, pix_width);
    cudaDeviceSynchronize();
    return d_curandstate;
}

__device__ float Sampler::pixCosineRng(unsigned int t_idx) {
    float uniform = curand_uniform(&d_curandstate[t_idx]);
    float transformed = asin(2*uniform-1);
    return transformed;
}
