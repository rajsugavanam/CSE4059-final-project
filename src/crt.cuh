#ifndef CRT_CUH
#define CRT_CUH

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
    // curand_init(1337, col, 0, &rngStates[col]);
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
__device__ Vec3 rayColor(const Ray& ray, Triangle3* triangles,
                         const int& num_triangle, Triangle3* wall_tri,
                         const int& num_wall_tri) {
    bool hit_anything = false;
    // check wall_mesh hit
    // int wall_tri_idx = closestTriangleIdx(ray, wall_tri, num_wall_tri);
    // Triangle3 hit_wall;
    // Vec3 wall_intersect;
    // float u_wall, v_wall, t_wall;
    // if (wall_tri_idx < num_wall_tri) {
    //     hit_wall = wall_tri[wall_tri_idx];
    //     rayIntersectsTriangle(ray, hit_wall, wall_intersect, u_wall, v_wall,
    //     t_wall); hit_anything = true;
    // }

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

    // check if wall or tri is closer
    Vec3 closest_normal;
    // if (t_wall < t) {
    //     hit_triangle = hit_wall;
    //     u = u_wall;
    //     v = v_wall;
    //     closest_normal = cross(hit_triangle.edge0(), hit_triangle.edge1());
    // }

    // Vec3 closest_normal;
    if (hit_anything) {
        // hit_triangle = triangles[triangle_idx];
        // float u, v, w, t;
        // rayIntersectsTriangle(ray, hit_triangle, closest_normal, u, v, t);

        // barycentric coord for face normal interpolation
        w = 1.0f - u - v;
        closest_normal = w * hit_triangle.normal0() +
                         u * hit_triangle.normal1() +
                         v * hit_triangle.normal2();

        // for (int diffuseIdx)
        // TODO: replace with "monte carlo" method for loop thing
        // simple lambertian diffuse
        Vec3 light = unit_vector(Vec3(0.0f, 5.0f, -5.0f) - tri_intersect);
        // just whiteCol color for now... use texture norm
        Vec3 whiteCol = Vec3(1.0f, 1.0f, 1.0f);
        float cos = dot(closest_normal, light);
        if (cos < 0.0f) {  // not sure if this is needed
            cos = 0.0f;
        }
        // return whiteCol * cos * 0.8f;

        // NORMAL MAP SAUCE
        closest_normal = unit_vector(closest_normal);
        return 0.5f * Vec3(closest_normal.x() + 1.0f, closest_normal.y() + 1.0f,
                           closest_normal.z() + 1.0f);
    } else {
        // Background color when no intersection
        Vec3 unit_direction = unit_vector(ray.direction());
        float alpha =
            0.5f * (unit_direction.y() + 1.0);  // y = [-1,1] to y = [0,1]
        // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
        return (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
               alpha * Vec3(0.5f, 0.7f, 1.0f);
    }
}

// TODO: use structs here
__global__ void rayRender(Vec3* image_buffer, int width, int height,
                          Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
                          Vec3 camera_origin, Triangle3* triangle_mesh,
                          int num_triangles, Triangle3* wall_mesh,
                          int num_wall_tri) {
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
        image_buffer[pixel_idx] = rayColor(ray, triangle_mesh, num_triangles,
                                           wall_mesh, num_wall_tri);
    }
}

// =====================================
// ===== OLD RENDER WITH TRIANGLES =====
// =====================================
// Möller–Trumbore intersection algorithm
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
// __device__ bool rayIntersectsTriangle(const Ray& ray, const Triangle3&
// triangle,
//                                       Vec3& intersection, float& u, float& v)
//                                       {
//     const float epsilon = 1.19209e-07f;
//
//     Vec3 edge1 = triangle.edge0();
//     Vec3 edge2 = triangle.edge1();
//     Vec3 ray_cross_e2 = cross(ray.direction(), edge2);
//     float determinant = dot(edge1, ray_cross_e2);
//
//     // for parallel to triangle
//     if (determinant > -epsilon && determinant < epsilon) {
//         return false;
//     }
//
//     float inv_determinant = 1.0f / determinant;
//     Vec3 s = ray.origin() - triangle.vertex0();
//     u = inv_determinant * dot(s, ray_cross_e2);
//
//     // no nuggies
//     if ((u < 0 && abs(u) > epsilon) || (u > 1 && abs(u - 1) > epsilon)) {
//         return false;
//     }
//
//     Vec3 s_cross_e1 = cross(s, edge1);
//     v = inv_determinant * dot(ray.direction(), s_cross_e1);
//
//     if ((v < 0 && abs(v) > epsilon) ||
//         (u + v > 1 && abs(u + v - 1) > epsilon)) {
//         return false;
//     }
//
//     // compute t to find interesection point
//     float t = inv_determinant * dot(edge2, s_cross_e1);
//
//     if (t > epsilon) {
//         intersection = ray.origin() + ray.direction() * t;
//         return true;
//     } else {
//         return false;
//     }
// }
//
// // alpha blending with precise method
// // normal map works for now... changing one z coord of triangle will
// correctly
// // map the color!
// __device__ Vec3 rayColor(const Ray& ray, Triangle3* triangles,
//                          int num_triangle) {
//     // Triangle3 triangle(Vec3(0.0f, 0.5f, -1.0f),     // top
//     //                    Vec3(0.5f, -0.5f, -1.0f),    // right
//     //                    Vec3(-0.5f, -0.5f, -1.0f));  // left
//     // Vec3 intersection;
//     Vec3 result{};
//     for (int tri = 0; tri < num_triangle; tri++) {
//         Triangle3 triangle = triangles[tri];
//         Vec3 intersection;
//         float u, v;
//         if (rayIntersectsTriangle(ray, triangle, intersection, u, v)) {  //
//         return Vec3(1.0f, 0.0f, 0.0f); //
//                                   // red for now... later we use
//             // Calculate interpolated normal using barycentric coordinates
//             float w = 1.0f - u - v; // Third barycentric coordinate
//
//             // Interpolate vertex normals
//             Vec3 interpolated_normal =
//                 w * triangle.vertexNormals[0] +
//                 u * triangle.vertexNormals[1] +
//                 v * triangle.vertexNormals[2];
//
//             // Normalize the interpolated normal
//             interpolated_normal = unit_vector(interpolated_normal);
//
//             // Convert normal to color
//             result = 0.5f * Vec3(interpolated_normal.x() + 1.0f,
//                                  interpolated_normal.y() + 1.0f,
//                                  interpolated_normal.z() + 1.0f);
//             break;
//         }
//         Vec3 unit_direction = unit_vector(ray.direction());
//         float alpha =
//             0.5f * (unit_direction.y() + 1.0);  // y = [-1,1] to y = [0,1]
//         // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
//         result = (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
//                  alpha * Vec3(0.5f, 0.7f, 1.0f);
//     }
//     return result;
// }
//
// __global__ void rayRender(Vec3* image_buffer, int width, int height,
//                           Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
//                           Vec3 camera_origin, Triangle3* triangle_mesh,
//                           int num_triangles) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     if (col < width && row < height) {
//         int pixel_idx = row * width + col;  // 3 channels (RGB)
//
//         Vec3 pixel_center = pixel00_loc + (col * delta_u) + (row * delta_v);
//         Vec3 ray_direction = pixel_center - camera_origin;
//
//         Ray ray(camera_origin, ray_direction);
//         Vec3 pixel_color = rayColor(ray, triangle_mesh, num_triangles);
//         image_buffer[pixel_idx] = pixel_color;
//     }
// }
//
// __global__ void greenRedRender(Vec3* image_buffer, int width, int height,
//                                Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
//                                Vec3 camera_origin) {
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     if (col < width && row < height) {
//         // int pixel_idx = (row * width + col) * 3;  // 3 channels (RGB)
//         // image_buffer[pixel_idx] = float(col) / float(width);       // Red
//         // image_buffer[pixel_idx + 1] = float(row) / float(height);  //
//         Green
//         // image_buffer[pixel_idx + 2] = 0.5f;                         //
//         Blue
//
//         // each thread computes all 3 channels
//         int pixel_idx = row * width + col;
//         image_buffer[pixel_idx] =
//             Vec3(float(col) / float(width), float(row) / float(height),
//             0.5f);
//     }
// }
#endif
