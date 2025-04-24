#ifndef RAY_COLOR_CUH
#define RAY_COLOR_CUH

#include "ray.cuh"
#include "vec3.cuh"

// Normal Map Color
__device__ __forceinline__ Vec3 normal_map(const Vec3& normal) {
    // Convert normal to color
    // Normal is in the range [-1, 1]
    // Map to [0, 1] range
    Vec3 temp = unit_vector(normal);
    float r = 0.5f * (temp.x() + 1.0f);
    float g = 0.5f * (temp.y() + 1.0f);
    float b = 0.5f * (temp.z() + 1.0f);
    return Vec3(r, g, b);
}

// Background color (gradient from blue to white)
__device__ __forceinline__ Vec3 sky_bg(const Ray& ray) {
    Vec3 unit_direction = unit_vector(ray.direction());
    float alpha =
        0.5f * (unit_direction.y() + 1.0f);  // y = [-1,1] to y = [0,1]
    // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
    return (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
           alpha * Vec3(0.5f, 0.7f, 1.0f);
}
#endif  // RAY_COLOR_CUH