#ifndef CUH_LIGHT_SAMPLING
#define CUH_LIGHT_SAMPLING

#include "ray.cuh"
#include "triangle3.cuh"
#include "vec3.cuh"
#include <curand_kernel.h>

// TODO: CHANGE ME (if needed)!!!!!!!
#define f_r 1

__global__ void kerInitRng(curandState* curand_state, int pix_height, int pix_width);
// WARNING: For testing only. Don't use!!
__global__ void kerCosineRng(float* values, curandState* curand_state, int pix_height, int pix_width);

struct ColoredRay {
    Vec3 color;
    Ray ray;
};

class Sampler {
    private:
        // const Vec3& input_ray, bary_hit_point;
        // const Triangle3* hit_tri;
        const int KERNEL_BLOCK_SIZE;
        curandState* d_curandstate;
        int pix_width, pix_height;
        int monte_carlo_samples;

        curandState* initRng();
        __device__ float pixCosineRng(unsigned int t_idx);
        // https://en.wikipedia.org/wiki/Rendering_equation
        // NOTE: here we are assuming f_r = constant. For now this is a macro.
        // colored_ray contains lambda and omega_i. (in essence)
        // bary_hit_point will be converted to x, given the triangle the coordinates are for.
        // omega_o is SAMPLED via Monte Carlo methods.
        __device__ float noLuminanceIntegral(const ColoredRay& colored_ray, const Vec3& bary_hit_point, Triangle3* hit_tri);

    public:
        Sampler(int kernel_block_size, int monte_carlo_samples, int pix_width, int pix_height);
        ~Sampler();
};

#endif
