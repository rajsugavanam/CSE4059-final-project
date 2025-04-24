#ifndef AABB_OLD_CUH
#define AABB_OLD_CUH

// #include <cuda_runtime.h>
#include "cuda_helper.h"
#include "reduction_old.cuh"
#include "triangle3.cuh"
#include "ray.cuh"
// #include "vec3.cuh"

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
                if (ray_origin[axis] < box_min[axis] || ray_origin[axis] > box_max[axis]) {
                    return false; // no intersection
                }
                continue; // go to next axis to avoid division by zero
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
#endif