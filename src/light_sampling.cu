#include "light_sampling.cuh"
#include <math.h>

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
    float transformed = asinf(2*uniform_value-1);
    values[t] = transformed;
}

// Sampler::Sampler(const Vec3& input_ray, const Triangle3* hit_tri,
//                  const Vec3& bary_hit_point)
//     : input_ray{input_ray}, hit_tri{hit_tri}, bary_hit_point{bary_hit_point} {}

Sampler::Sampler(int kernel_block_size, int monte_carlo_samples, int pix_width, int pix_height)
    : KERNEL_BLOCK_SIZE(kernel_block_size), monte_carlo_samples(monte_carlo_samples),
      pix_width(pix_width), pix_height(pix_height) {}

Sampler::~Sampler() {
    if (d_curandstate != nullptr) {
        cudaFree(d_curandstate); // RAII?
    }
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
    // https://en.wikipedia.org/wiki/Inverse_transform_sampling
    float uniform = curand_uniform(&d_curandstate[t_idx]);
    float transformed = asin(2*uniform-1);
    return transformed;
}

__device__ void Sampler::setGPUInputs(TriangleMesh* d_mesh, const AABB* boxes, int num_objects, const int* num_triangles) {
    this->d_mesh = d_mesh;
    this->boxes = boxes;
    this->num_objects = num_objects;
    this->num_triangles = num_triangles;
}

__device__ Vec3 Sampler::noLuminanceIntegral(const Vec3& intersection_point) {
    // monte carlo formula = (1/N) \sum_{i=1}^N f(x)/pdf(x) = \int_{\Omega} f(x) dx.
    // NOTE: symbol conventions according to https://en.wikipedia.org/wiki/Monte_Carlo_integration.
    // assume radiance is roughly equal to intensity.
    Vec3 I(0.0f,0.0f,0.0f);
    float V = 2.0f*M_PI/3.0f; // integral of input omega.

    unsigned int t = blockDim.x*blockIdx.x + threadIdx.x;

    for (int i=0; i<monte_carlo_samples; i++) {
        float cosine = pixCosineRng(t);
        float theta = curand_uniform(&d_curandstate[t]);
        float rho = acosf(cosine);
        float horiz_mag = __sinf(rho);

        float x = horiz_mag*__cosf(theta);
        float z = horiz_mag*__sinf(theta);

        // TODO: get color as a function of omega_i. this will consider omega_i's bounce and, if it hits another
        // triangle, recursively calls noLuminanceIntegral, using its return value as omega_i's color. This will
        // be added to the integrand I.

        Vec3 omega_i_vec(x, cosine, z);
        Ray omega_i(
            intersection_point,
            omega_i_vec
        );
        Vec3 omega_i_result_color = colorPathTrace(omega_i);
        // add integrand.
        I += f_r*omega_i_result_color;

    }

    I = I*(V/((float)monte_carlo_samples));
    return I;
}

__device__ Vec3 Sampler::colorPathTrace(const Ray& omega_i) {
}
