#ifndef FIRST_CRT_CUH
#define FIRST_CRT_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>

#include "ray.cuh"
#include "triangle3.cuh"
#include "vec3.cuh"

// coalesced init for RNG
__global__ void initRng(curandState* rngStates, unsigned int width) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int pixel_idx = row * width + col;
    // you just have to call to initialize the RNG
    curand_init(1337, pixel_idx, 0, &rngStates[pixel_idx]);
}

// Möller–Trumbore intersection algorithm
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool rayIntersectsTriangle(const Ray& ray, const Triangle3& triangle,
                                      Vec3& intersection, float& u, float& v,
                                      float& t) {
    const float epsilon = 1.19209e-07f;  // float32 machine epsilon ish

    Vec3 edge1 = triangle.edge0();
    Vec3 edge2 = triangle.edge1();
    Vec3 ray_cross_e2 = cross(ray.direction(), edge2);
    float determinant = dot(edge1, ray_cross_e2);

    // Parallel to triangle
    if (fabsf(determinant) < epsilon) {
        return false;
    }

    float inv_determinant = 1.0f / determinant;
    Vec3 s = ray.origin() - triangle.vertex0();
    u = inv_determinant * dot(s, ray_cross_e2);

    // No nuggies
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    Vec3 s_cross_e1 = cross(s, edge1);
    v = inv_determinant * dot(ray.direction(), s_cross_e1);

    // no nuggies pt 2
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    // compute t to find intersection point
    t = inv_determinant * dot(edge2, s_cross_e1);

    if (t > epsilon) {
        intersection = ray.origin() + ray.direction() * t;
        return true;
    }
    return false;
}

// find closest intersecting triangle within a mesh
__device__ int closestTriangleIdx(const Ray& ray,
                                  const Triangle3* triangle_mesh,
                                  int num_triangles) {
    Triangle3 curr_tri;
    float closest_distance = INFINITY;
    int closest_triangle_idx = num_triangles;  // nonexisting tri for no hit;
    Vec3 intersection;
    float u, v, t;

    for (int tri_idx = 0; tri_idx < num_triangles; tri_idx++) {
        curr_tri = triangle_mesh[tri_idx];
        if (rayIntersectsTriangle(ray, curr_tri, intersection, u, v, t)) {
            // t = -intersection.z();
            if (t < closest_distance) {
                closest_distance = t;
                closest_triangle_idx = tri_idx;
            }
        }
    }
    return closest_triangle_idx;
}
// alpha blending with precise method
// normal map works for now... changing one z coord of triangle will correctly
// map the color!
__device__ Vec3 colorRay(const Ray& ray, Triangle3* triangles,
                         const int& num_triangle) {
    bool hit_anything = false;

    // check triangle_mesh hit
    int triangle_idx = closestTriangleIdx(ray, triangles, num_triangle);
    Triangle3 hit_triangle;
    Vec3 tri_intersect;
    float u;
    float v;
    float w;
    float t = INFINITY;

    if (triangle_idx < num_triangle) {
        hit_triangle = triangles[triangle_idx];
        rayIntersectsTriangle(ray, hit_triangle, tri_intersect, u, v, t);
        hit_anything = true;
    }


    if (hit_anything) {
        // hit_triangle = triangles[triangle_idx];
        // float u, v, w, t;
        // rayIntersectsTriangle(ray, hit_triangle, closest_normal, u, v, t);

        Vec3 closest_normal;

        // barycentric coord for face normal interpolation
        w = 1.0f - u - v;
        closest_normal = w * hit_triangle.normal0() +
                         u * hit_triangle.normal1() +
                         v * hit_triangle.normal2();

        // for (int diffuseIdx)
        // simple lambertian diffuse
        // Vec3 light = unit_vector(Vec3(0.0f, 5.0f, -5.0f) - tri_intersect);
        // // just whiteCol color for now... use texture norm
        // Vec3 whiteCol = Vec3(1.0f, 1.0f, 1.0f);
        // float cos = dot(closest_normal, light);
        // if (cos < 0.0f) {  // not sure if this is needed
        //     cos = 0.0f;
        // }
        // return whiteCol * cos * 0.8f;

        // NORMAL MAP SAUCE
        closest_normal = unit_vector(closest_normal);
        return 0.5f * Vec3(closest_normal.x() + 1.0f, closest_normal.y() + 1.0f,
                           closest_normal.z() + 1.0f);
    } else {
        return Vec3(0.0f, 0.0f, 0.0f);  // black background color
    }
}

__global__ void rayRender(Vec3* image_buffer, int width, int height,
                          Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
                          Vec3 camera_origin, Triangle3* triangle_mesh,
                          int num_triangles) {
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
        image_buffer[pixel_idx] = colorRay(ray, triangle_mesh, num_triangles);
    }
}
#endif
