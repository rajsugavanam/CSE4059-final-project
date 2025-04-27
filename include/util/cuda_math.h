#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <cuda_runtime.h>

// Vector utility functions using CUDA vector types
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator*(float b, const float3& a) {
    return a * b;  // Reuse the other operator* implementation
}

__host__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ float length_squared(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__host__ __device__ float length(float3 a) {
    return sqrtf(length_squared(a));
}

__host__ __device__ float3 normalize(float3 a) {
    float rlength = rsqrtf(length_squared(a));
    return make_float3(a.x * rlength, a.y * rlength, a.z * rlength);
}

#endif // CUDA_MATH_H