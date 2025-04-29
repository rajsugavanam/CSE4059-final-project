#ifndef RAY_COLOR_CUH
#define RAY_COLOR_CUH

#include "ray.cuh"
#include "vec3.cuh"

// Normal Map Color
__device__ __forceinline__ Vec3 normalMap(const Vec3& normal) {
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
__device__ __forceinline__ Vec3 skyBg(const Ray& ray) {
    Vec3 unit_direction = unit_vector(ray.direction());
    float alpha =
        0.5f * (unit_direction.y() + 1.0f);  // y = [-1,1] to y = [0,1]
    // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
    return (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
           alpha * Vec3(0.5f, 0.7f, 1.0f);
}

__device__ __forceinline__ Vec3 threeColor(int hit_obj_id) {
    // Simple coloring based on object ID and triangle index
    switch (hit_obj_id % 3) {
        case 0:
            return Vec3(1.0f, 0.2f, 0.2f);  // Red
        case 1:
            return Vec3(0.2f, 1.0f, 0.2f);  // Green
        default:
            return Vec3(0.2f, 0.2f, 1.0f);  // Blue
    }
}

__device__ __forceinline__ Vec3 quantizedColor(int hit_obj_id) {
    // Cool OKLAB perceptual uniform color gradient (Palette 3)
    // https://evannorton.github.io/Acerolas-Epic-Color-Palettes/
    float3 colors[8] = {
        make_float3(60.0f, 34.0f, 74.0f),
        make_float3(89.0f, 45.0f, 83.0f),
        make_float3(122.0f, 58.0f, 91.0f),
        make_float3(159.0f, 75.0f, 97.0f),
        make_float3(199.0f, 96.0f, 104.0f),
        make_float3(240.0f, 122.0f, 110.0f),
        make_float3(255.0f, 155.0f, 119.0f),
        make_float3(255.0f, 195.0f, 132.0f)
    };
    return Vec3(colors[hit_obj_id % 8].x / 255.0f,
                colors[hit_obj_id % 8].y / 255.0f,
                colors[hit_obj_id % 8].z / 255.0f);
}

#endif  // RAY_COLOR_CUH