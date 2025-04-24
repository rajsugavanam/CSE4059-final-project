#ifndef CUH_LIGHT_SAMPLING
#define CUH_LIGHT_SAMPLING

#include "triangle3.cuh"
#include "vec3.cuh"
#include <curand_kernel.h>

__global__ void kerInitRng(curandState* curand_state, int pix_height, int pix_width);
// WARNING: For testing only. Don't use!!
__global__ void kerCosineRng(float* values, curandState* curand_state, int pix_height, int pix_width);

class Sampler {
    private:
        // const Vec3& input_ray, bary_hit_point;
        // const Triangle3* hit_tri;
        const int KERNEL_BLOCK_SIZE;
        curandState* d_curandstate;
        int pix_width, pix_height;

        curandState* initRng();
        __device__ float pixCosineRng(unsigned int t_idx);

    public:
        Sampler(int kernel_block_size, int pix_width, int pix_height);
        ~Sampler();
};

#endif
