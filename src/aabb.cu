#include "aabb.cuh"

// Constructor implementation
__host__ AABB::AABB()
    : num_obj{0},
      h_minx{nullptr},
      h_miny{nullptr},
      h_minz{nullptr},
      h_maxx{nullptr},
      h_maxy{nullptr},
      h_maxz{nullptr},
      d_minx{nullptr},
      d_miny{nullptr},
      d_minz{nullptr},
      d_maxx{nullptr},
      d_maxy{nullptr},
      d_maxz{nullptr} {}

__host__ AABB::AABB(int num_obj) : num_obj(num_obj) {
    mallocAABB();
    cudaMallocAABB();
}

__host__ AABB::~AABB() {
    freeAABB();
    cudaFreeAABB();
    // std::cout << "AABB destructor called" << std::endl;
}

__host__ void AABB::mallocAABB() {
    h_minx = new float[num_obj];
    h_miny = new float[num_obj];
    h_minz = new float[num_obj];
    h_maxx = new float[num_obj];
    h_maxy = new float[num_obj];
    h_maxz = new float[num_obj];
}

__host__ void AABB::freeAABB() {
    delete[] h_minx;
    delete[] h_miny;
    delete[] h_minz;
    delete[] h_maxx;
    delete[] h_maxy;
    delete[] h_maxz;

    h_minx = nullptr;
    h_miny = nullptr;
    h_minz = nullptr;
    h_maxx = nullptr;
    h_maxy = nullptr;
    h_maxz = nullptr;
}

__host__ void AABB::cudaMallocAABB() {
    cudaMalloc((void**)&d_minx, num_obj * sizeof(float));
    cudaMalloc((void**)&d_miny, num_obj * sizeof(float));
    cudaMalloc((void**)&d_minz, num_obj * sizeof(float));
    cudaMalloc((void**)&d_maxx, num_obj * sizeof(float));
    cudaMalloc((void**)&d_maxy, num_obj * sizeof(float));
    cudaMalloc((void**)&d_maxz, num_obj * sizeof(float));
}

__host__ void AABB::AABBMemcpyHtD() {
    cudaMemcpy(d_minx, h_minx, num_obj * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_miny, h_miny, num_obj * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_minz, h_minz, num_obj * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxx, h_maxx, num_obj * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxy, h_maxy, num_obj * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxz, h_maxz, num_obj * sizeof(float),
               cudaMemcpyHostToDevice);
}

__host__ void AABB::AABBMemcpyDtH() {
    cudaMemcpy(h_minx, d_minx, num_obj * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_miny, d_miny, num_obj * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_minz, d_minz, num_obj * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxx, d_maxx, num_obj * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxy, d_maxy, num_obj * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxz, d_maxz, num_obj * sizeof(float),
               cudaMemcpyDeviceToHost);
}

__host__ void AABB::cudaFreeAABB() {
    cudaFree(d_minx);
    cudaFree(d_miny);
    cudaFree(d_minz);
    cudaFree(d_maxx);
    cudaFree(d_maxy);
    cudaFree(d_maxz);
}

// SOURCE: IRT (p.65)- An Introduction to Ray Tracing, Andrew S. Glassner
// https://education.siggraph.org/static/HyperGraph/raytrace/rtinter3.html
__device__ bool AABB::hitAABB(const Ray&  ray, int idx) const {
    Vec3 ray_origin = ray.origin();
    Vec3 ray_direction = ray.direction();

    float t_near = -INFINITY;
    float t_far = INFINITY;

    // For near-zero direction (parallel rays)
    const float epsilon = 1e-8f;

    // Using pointer arithmetic to access min/max values to avoid branching
    // in the loop
    float* min_vals[3] = {d_minx + idx, d_miny + idx, d_minz + idx};
    float* max_vals[3] = {d_maxx + idx, d_maxy + idx, d_maxz + idx};

    // Loop over the three axes (x, y, z)
    for (int axis = 0; axis < 3; axis++) {
        float min_val = *min_vals[axis];
        float max_val = *max_vals[axis];
        float ray_dir = ray_direction[axis];
        float ray_orig = ray_origin[axis];

        // Special handling for near-zero directions (parallel rays)
        if (fabsf(ray_dir) < epsilon) {
            // If ray is parallel to axis and outside slab, no intersection
            if (ray_orig < min_val || ray_orig > max_val) {
                return false;
            }
            continue;  // Skip to next axis
        }

        // Calculate intersection times with the slab planes
        float inv_dir = 1.0f / ray_dir;
        float t0 = (min_val - ray_orig) * inv_dir;
        float t1 = (max_val - ray_orig) * inv_dir;

        // Ensure t0 <= t1
        if (t0 > t1) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        // Update intersection interval
        t_near = fmaxf(t_near, t0);
        t_far = fminf(t_far, t1);

        // Early exit if intervals are disjoint (ray misses box)
        // or if t_far is negative (box is behind the ray)
        if (t_near > t_far || t_far < 0) {
            return false;
        }
    }

    // If we get here, the ray intersects the AABB
    return true;
}

// Loop-based implementation of AABB intersection test
// Its actually slower than the pointer arithmetic version...
__device__ bool AABB::hitAABBLoop(const Ray& ray, int idx) const {
    Vec3 ray_origin = ray.origin();
    Vec3 ray_direction = ray.direction();

    float t_near = -INFINITY;
    float t_far = INFINITY;

    // Loop over the three axes (x, y, z)
    for (int axis = 0; axis < 3; axis++) {
        float min_val, max_val;
        // float ray_ori = ray_origin[axis];
        float ray_dir = ray_direction[axis];

        // Select the appropriate min/max values based on axis
        if (axis == 0) {
            min_val = d_minx[idx];
            max_val = d_maxx[idx];
        } else if (axis == 1) {
            min_val = d_miny[idx];
            max_val = d_maxy[idx];
        } else {  // axis == 2
            min_val = d_minz[idx];
            max_val = d_maxz[idx];
        }

        // Handle parallel rays
        if (ray_dir == 0) {
            if (ray_origin[axis] < min_val || ray_origin[axis] > max_val) {
                return false;  // Ray is parallel to the slab and outside it
            }
        } else {
            // Calculate intersection times with the slab planes
            float t0 = (min_val - ray_origin[axis]) / ray_dir;
            float t1 = (max_val - ray_origin[axis]) / ray_dir;

            // Ensure t0 <= t1
            if (t0 > t1) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }

            // Update t_near and t_far
            if (t0 > t_near) t_near = t0;
            if (t1 < t_far) t_far = t1;

            // Check for no intersection
            if (t_near > t_far || t_far < 0) return false;
        }
    }

    // If we got here, the ray intersects all three slabs
    return true;
}
