#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// #include "aabb_old.cuh"
#include "camera.h"
#include "cuda_helper.h"
#include "image_writer.h"
#include "obj_reader.cuh"
#include "ray.cuh"
#include "timer.h"
#include "triangle3.cuh"
#include "vec3.cuh"

// reduction_old.cuh moved from include to separate test from main
__device__ float atomicMinf(float* address, float val);
__device__ float atomicMaxf(float* address, float val);
__global__ void minReduceAtomic(const float* input, float* output, int size,
                                float identity);
__global__ void maxReduceAtomic(const float* input, float* output, int size,
                                float identity);

__global__ void minTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity);

__global__ void maxTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity);

// aabb_old.cuh placed here for testing
class AABB {
   public:
    Vec3 box_min;
    Vec3 box_max;

    __host__ __device__ AABB() {
        box_min = Vec3(INFINITY, INFINITY, INFINITY);
        box_max = Vec3(-INFINITY, -INFINITY, -INFINITY);
    }

    __host__ __device__ AABB(Vec3 min, Vec3 max) {
        box_min = min;
        box_max = max;
    }

    // SOURCE: IRT (p.65)- An Introduction to Ray Tracing, Andrew S. Glassner
    // https://education.siggraph.org/static/HyperGraph/raytrace/rtinter3.htm
    __host__ __device__ bool hitAABB(Ray ray) const {
        // Ray P = O + tD
        Vec3 ray_origin = ray.origin();
        Vec3 ray_direction = ray.direction();

        // set t_near and t_far to infty
        float t_near = -INFINITY;
        float t_far = INFINITY;

        // for each pair of planes do
        for (int axis = 0; axis < 3; axis++) {
            // if ray is parallel to slab
            if (ray_direction[axis] == 0) {
                // if ray origin is outside slab
                if (ray_origin[axis] < box_min[axis] ||
                    ray_origin[axis] > box_max[axis]) {
                    return false;  // no intersection
                }
                continue;  // go to next axis to avoid division by zero
            }

            // calculate t_near and t_far
            // t0 = (x0 - Ox) / Dx
            float t0 = (box_min[axis] - ray_origin[axis]) / ray_direction[axis];
            float t1 = (box_max[axis] - ray_origin[axis]) / ray_direction[axis];

            // swap t0 and t1 if necessary
            if (t0 > t1) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            // update t_near and t_far
            if (t0 > t_near) {
                t_near = t0;
            }
            if (t1 < t_far) {
                t_far = t1;
            }

            // if the intervals are disjoint or slab is missed
            if (t_near > t_far || t_far < 0) {
                return false;
            }
        }
        // ray intersects all three slabs
        return true;
    }
};

// perhaps move all this to a struct...
const float aspect_ratio = 16.0f / 9.0f;
const int pixel_height = 1080;
const int pixel_width = static_cast<int>(pixel_height * aspect_ratio);
const float focal_length = 1.0f;
const float viewport_height = 2.0f;
const float viewport_width =
    viewport_height * (float(pixel_width) / pixel_height);
const float camera_x = 0.0f;
const float camera_y = 0.0f;
const float camera_z = 0.0f;
// const Vec3 camera_center = Vec3(0.0f, -6.0f, 7.0f); // the sphere is at a
// weird location
const Vec3 camera_center = Vec3(camera_x, camera_y, camera_z);

// viewport vector
const Vec3 viewport_u = Vec3(viewport_width, 0.0f, 0.0f);
const Vec3 viewport_v = Vec3(0.0f, -viewport_height, 0.0f);

const Vec3 pixel_delta_u = viewport_u / pixel_width;
const Vec3 pixel_delta_v = viewport_v / pixel_height;

const Vec3 viewport_upper_left = camera_center -
                                 Vec3(0.0f, 0.0f, focal_length) -
                                 viewport_u / 2 - viewport_v / 2;
const Vec3 pixel00_loc =
    viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
const int image_buffer_size = pixel_width * pixel_height;
const size_t image_buffer_byte_size = image_buffer_size * sizeof(Vec3);

__device__ Vec3 colorRay(const Ray& ray, AABB* box) {
    // Check intersection with box
    if (box->hitAABB(ray)) {
        // If ray hits box, color the pixel red
        return Vec3(1.0f, 0.0f, 0.0f);
    } else {
        // Background color (gradient from blue to white) - exactly like in
        // include/first_crt.cuh
        Vec3 unit_direction = unit_vector(ray.direction());
        float alpha =
            0.5f * (unit_direction.y() + 1.0f);  // y = [-1,1] to y = [0,1]
        // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
        return (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
               alpha * Vec3(0.5f, 0.7f, 1.0f);
    }
}

// Rewritten to match exactly the style of rayRender in include/first_crt.cuh
__global__ void renderBoxKernel(Vec3* image_buffer, int width, int height,
                                Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
                                Vec3 camera_origin, AABB* box) {
    // Calculate pixel coordinates
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        const int pixel_idx = row * width + col;

        // ray params
        const Vec3 pixel_center =
            pixel00_loc + (col * delta_u) + (row * delta_v);
        const Vec3 ray_direction = pixel_center - camera_origin;

        Ray ray(camera_origin, ray_direction);
        image_buffer[pixel_idx] = colorRay(ray, box);
    }
}

// Add a CPU function to compute AABB bounds for comparison
void computeAABBHost(const std::vector<Triangle3>& triangles, Vec3& min_bounds,
                     Vec3& max_bounds) {
    if (triangles.empty()) return;

    // Initialize bounds with first vertex of first triangle
    min_bounds = triangles[0].vertex0();
    max_bounds = triangles[0].vertex0();

    // Check all vertices of all triangles
    for (const auto& triangle : triangles) {
        // Check vertex 0
        min_bounds = Vec3(fminf(min_bounds.x(), triangle.vertex0().x()),
                          fminf(min_bounds.y(), triangle.vertex0().y()),
                          fminf(min_bounds.z(), triangle.vertex0().z()));
        max_bounds = Vec3(fmaxf(max_bounds.x(), triangle.vertex0().x()),
                          fmaxf(max_bounds.y(), triangle.vertex0().y()),
                          fmaxf(max_bounds.z(), triangle.vertex0().z()));

        // Check vertex 1
        min_bounds = Vec3(fminf(min_bounds.x(), triangle.vertex1().x()),
                          fminf(min_bounds.y(), triangle.vertex1().y()),
                          fminf(min_bounds.z(), triangle.vertex1().z()));
        max_bounds = Vec3(fmaxf(max_bounds.x(), triangle.vertex1().x()),
                          fmaxf(max_bounds.y(), triangle.vertex1().y()),
                          fmaxf(max_bounds.z(), triangle.vertex1().z()));

        // Check vertex 2
        min_bounds = Vec3(fminf(min_bounds.x(), triangle.vertex2().x()),
                          fminf(min_bounds.y(), triangle.vertex2().y()),
                          fminf(min_bounds.z(), triangle.vertex2().z()));
        max_bounds = Vec3(fmaxf(max_bounds.x(), triangle.vertex2().x()),
                          fmaxf(max_bounds.y(), triangle.vertex2().y()),
                          fmaxf(max_bounds.z(), triangle.vertex2().z()));
    }
}

// This test is to check the AABB reduction algorithm
int oldReduction() {
    // ===================
    // ===== THE OBJ =====
    // ===================
    Timer timer;
    timer.start("Loading OBJ file");
    ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/dk.obj");
    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) +
    // "/assets/large_sphere.obj");
    reader.readModel();
    Model model = reader.parsedModel;
    assert(!model.modelTriangles.empty());
    timer.stop();

    Vec3* image_buffer_h{new Vec3[image_buffer_size]};
    Vec3* image_buffer_d;

    // load all the triangles
    Triangle3* triangles_h = model.modelTriangles.data();
    Triangle3* triangles_d;
    size_t mesh_size = model.modelTriangles.size();
    size_t mesh_byte_size = mesh_size * sizeof(Triangle3);
    std::cout << "OBJ MODEL SIZE: " << mesh_size << "\n";

    // malloc state
    cudaMalloc((void**)&image_buffer_d, image_buffer_byte_size);
    cudaMalloc((void**)&triangles_d, mesh_byte_size);
    cudaMemcpy(image_buffer_d, image_buffer_h, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(triangles_d, triangles_h, mesh_byte_size,
               cudaMemcpyHostToDevice);
    timer.stop();

    // ===========================
    // ===== RUNNING ON GPU ======
    // ===========================

    // INIT AABB
    AABB* d_aabb = nullptr;
    AABB* h_aabb = new AABB[1]{};
    size_t aabb_size = sizeof(AABB);

    // Initialize with extreme values to ensure proper min/max calculation
    float* h_aabb_min = new float[3]{INFINITY, INFINITY, INFINITY};
    float* h_aabb_max = new float[3]{-INFINITY, -INFINITY, -INFINITY};
    float* d_aabb_min = nullptr;
    float* d_aabb_max = nullptr;

    cudaMalloc((void**)&d_aabb, aabb_size);
    cudaMalloc((void**)&d_aabb_min, 3 * sizeof(float));
    cudaMalloc((void**)&d_aabb_max, 3 * sizeof(float));
    cudaMemcpy(d_aabb_min, h_aabb_min, 3 * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_aabb_max, h_aabb_max, 3 * sizeof(float),
               cudaMemcpyHostToDevice);

    // REDUCTION
    dim3 aabb_block_size(256);
    dim3 aabb_grid_size((mesh_size + aabb_block_size.x - 1) /
                        aabb_block_size.x);
    timer.start("AABB reduction");

    // Allocate enough shared memory for 3 arrays of floats (min/max x, y, z)
    size_t sharedMemSize = 3 * aabb_block_size.x * sizeof(float);

    minTriangleMesh<<<aabb_grid_size, aabb_block_size, sharedMemSize>>>(
        triangles_d, mesh_size, d_aabb_min, INFINITY);
    maxTriangleMesh<<<aabb_grid_size, aabb_block_size, sharedMemSize>>>(
        triangles_d, mesh_size, d_aabb_max, -INFINITY);
    cudaDeviceSynchronize();
    timer.stop();

    // copy back to host
    cudaMemcpy(h_aabb_min, d_aabb_min, 3 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aabb_max, d_aabb_max, 3 * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compute AABB CPU Version
    timer.start("CPU AABB computation");
    Vec3 cpu_min_bounds, cpu_max_bounds;
    computeAABBHost(model.modelTriangles, cpu_min_bounds, cpu_max_bounds);
    timer.stop();

    // CPU vs GPU AABB comparison
    bool same_aabb = false;
    float epsilon = 1e-5f;

    if (std::abs(h_aabb_min[0] - cpu_min_bounds.x()) > epsilon ||
        std::abs(h_aabb_min[1] - cpu_min_bounds.y()) > epsilon ||
        std::abs(h_aabb_min[2] - cpu_min_bounds.z()) > epsilon ||
        std::abs(h_aabb_max[0] - cpu_max_bounds.x()) > epsilon ||
        std::abs(h_aabb_max[1] - cpu_max_bounds.y()) > epsilon ||
        std::abs(h_aabb_max[2] - cpu_max_bounds.z()) > epsilon) {
        same_aabb = true;
    }

    // Copy AABB data to device
    h_aabb[0].box_min = Vec3(h_aabb_min[0], h_aabb_min[1], h_aabb_min[2]);
    h_aabb[0].box_max = Vec3(h_aabb_max[0], h_aabb_max[1], h_aabb_max[2]);
    cudaMemcpy(d_aabb, h_aabb, aabb_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((pixel_width + block_size.x - 1) / block_size.x,
                   (pixel_height + block_size.y - 1) / block_size.y);

    // ======================
    // ===== RENDERING ======
    // ======================
    timer.start("Ray rendering");
    renderBoxKernel<<<grid_size, block_size>>>(
        image_buffer_d, pixel_width, pixel_height, pixel00_loc, pixel_delta_u,
        pixel_delta_v, camera_center, d_aabb);

    cudaDeviceSynchronize();
    timer.stop();

    // ============================
    // ===== CPU IMAGE WRITER =====
    // ============================
    timer.start("Copying back to host");
    cudaMemcpy(image_buffer_h, image_buffer_d, image_buffer_byte_size,
               cudaMemcpyDeviceToHost);
    timer.stop();

    timer.start("Outputting to PPM file");
    writeToPPM("obj_reduction_test.ppm", image_buffer_h, pixel_width,
               pixel_height);
    timer.stop();

    // print AABB from GPU reduction
    std::cout << "GPU AABB min: " << h_aabb_min[0] << ", " << h_aabb_min[1]
              << ", " << h_aabb_min[2] << "\n";
    std::cout << "GPU AABB max: " << h_aabb_max[0] << ", " << h_aabb_max[1]
              << ", " << h_aabb_max[2] << "\n";

    // Print CPU AABB results
    std::cout << "CPU AABB min: " << cpu_min_bounds.x() << ", "
              << cpu_min_bounds.y() << ", " << cpu_min_bounds.z() << "\n";
    std::cout << "CPU AABB max: " << cpu_max_bounds.x() << ", "
              << cpu_max_bounds.y() << ", " << cpu_max_bounds.z() << "\n";

    // ==========================
    // ===== MEMORY FREEDOM =====
    // ==========================
    delete[] image_buffer_h;
    delete[] h_aabb;
    delete[] h_aabb_min;
    delete[] h_aabb_max;
    cudaFree(image_buffer_d);
    cudaFree(triangles_d);
    cudaFree(d_aabb);
    cudaFree(d_aabb_min);
    cudaFree(d_aabb_max);

    if (same_aabb) {
        std::cerr << "AABB reduction test failed!" << std::endl;
        return 1;
    } else {
        std::cout << "AABB reduction test passed!" << std::endl;
        return 0;
    }
}

int main(int argc, char* argv[]) { return oldReduction(); }

// reduction_old.cuh below
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

__global__ void minTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_minf, float identity) {
    // Use dynamic shared memory with proper offsets
    extern __shared__ float shared_mem[];
    float* s_minx = shared_mem;
    float* s_miny = &shared_mem[blockDim.x];
    float* s_minz = &shared_mem[2 * blockDim.x];

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

__global__ void maxTriangleMesh(const Triangle3* triangles, int num_tri,
                                float* box_maxf, float identity) {
    // Use dynamic shared memory with proper offsets
    extern __shared__ float shared_mem[];
    float* s_maxx = shared_mem;
    float* s_maxy = &shared_mem[blockDim.x];
    float* s_maxz = &shared_mem[2 * blockDim.x];

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