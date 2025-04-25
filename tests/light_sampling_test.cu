#include <cassert>
#include "light_sampling.cuh"
#include "camera.h"
#include "timer.h"
#include "obj_reader.cuh"
#include "include/crt2.cuh"

int main() {

    // =========== READ MODEL ===========
    // ==================================
    Timer timer;
    timer.start("Loading OBJ file.");

    ObjReader reader =
        ObjReader(std::string(PROJECT_ROOT) + "/assets/dk.obj");
    reader.readModel();
    Model model = reader.parsedModel;
    
    assert(!model.modelTriangles.empty());
    timer.stop();
    // ==================================
    // ==================================

    Camera camera(cornell_box_params);
    camera.CUDAparams();
    CUDACameraParams cuda_params = camera.CUDAparams();
    int pix_height = cuda_params.pixel_height;
    int pix_width = cuda_params.pixel_width;
    size_t img_pixels = pix_height*pix_width;
    size_t img_bytes = pix_height*pix_width*sizeof(Vec3);

    // one Vec3/pixel.
    Vec3* h_image_buffer{new Vec3[img_pixels]};
    Vec3* d_image_buffer;

    // don't malloc here, as we'll fail to free memory if a model isn't read.

    Triangle3* h_triangle_mesh = model.modelTriangles.data();
    Triangle3* d_triangle_mesh;
    size_t mesh_size = model.modelTriangles.size();
    size_t mesh_bytes = mesh_size * sizeof(Triangle3);
    std::cout << "OBJ MODEL SIZE: " << mesh_size << "\n";

    curandState* curand_state;
    float* d_rng_values;

    // ====== MEM ALLOCATION/COPY =======
    cudaMalloc(&d_triangle_mesh, mesh_bytes);
    cudaMalloc(&d_image_buffer, img_bytes);
    cudaMalloc(&curand_state, sizeof(curandState)*img_pixels);
    cudaMalloc(&d_rng_values, sizeof(float)*img_pixels);

    cudaMemcpy(d_triangle_mesh, h_triangle_mesh, mesh_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_buffer, h_image_buffer, img_bytes, cudaMemcpyHostToDevice);
    // no copy for curand_state!


    // ========== CREATE RAYS ===========
    const int size_2d = 16;
    const dim3 block_dim(size_2d, size_2d, 1);
    const dim3 grid_dim( (pix_width + size_2d - 1)/size_2d, (pix_height + size_2d - 1)/size_2d, 1 );
    rayRender<<<grid_dim, block_dim>>>(d_image_buffer, d_triangle_mesh, mesh_size, cuda_params);

    // ========== CREATE RNG ============
    kerInitRng<<<grid_dim, block_dim>>>(curand_state, pix_height, pix_width);
    cudaDeviceSynchronize();
    kerCosineRng<<<grid_dim, block_dim>>>(d_rng_values, curand_state, pix_height, pix_width);
    cudaDeviceSynchronize();

    // WARNING: temporary!
    // ======= GET SOME RESULTS =========
    float* h_rng_values{new float[img_pixels]};
    cudaMemcpy(h_rng_values, d_rng_values, img_pixels, cudaMemcpyDeviceToHost);
    std::cout << "Pixels: " << img_pixels << '\n';
    float sum = 0;
    float min = INFINITY;
    float max = -INFINITY;
    for (int i=0; i<img_pixels; i++) {
        float value = h_rng_values[i];
        sum += value;
        if (min > value) {
            min = value;
        }
        if (max < value) {
            max = value;
        }
    }
    std::cout << "AVG: " << sum/((float)img_pixels) << std::endl;
    std::cout << "MIN: " << min << std::endl;
    std::cout << "MAX: " << max << std::endl;


    // =========== MEM FREE =============
    free(h_rng_values);
    cudaFree(d_triangle_mesh);
    cudaFree(d_image_buffer);
}
