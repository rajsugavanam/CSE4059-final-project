#ifndef VEC3_CUH
#define VEC3_CUH

#include <iostream>

// maybe change to template for double precision
class Vec3 {
   public:
    float e_[3];

    // Constructors
    __host__ __device__ Vec3() : e_{0.0f, 0.0f, 0.0f} {}
    __host__ __device__ Vec3(float e0, float e1, float e2) : e_{e0, e1, e2} {}

    // Accessors
    __host__ __device__ __forceinline__ float x() const { return e_[0]; }
    __host__ __device__ __forceinline__ float y() const { return e_[1]; }
    __host__ __device__ __forceinline__ float z() const { return e_[2]; }

    // Operator overloads
    __host__ __device__ __forceinline__ float operator[](int idx) const { return e_[idx]; }
    __host__ __device__ __forceinline__ float& operator[](int idx) { return e_[idx]; }
    __host__ __device__ __forceinline__ Vec3& operator+=(const Vec3& vec3) {
        e_[0] += vec3.e_[0];
        e_[1] += vec3.e_[1];
        e_[2] += vec3.e_[2];
        return *this;
    }
    __host__ __device__ __forceinline__ Vec3& operator*=(float scalar) {
        e_[0] *= scalar;
        e_[1] *= scalar;
        e_[2] *= scalar;
        return *this;
    }
    __host__ __device__ __forceinline__ Vec3& operator/=(float scalar) {
        e_[0] *= 1.0f / scalar;
        e_[1] *= 1.0f / scalar;
        e_[2] *= 1.0f / scalar;
        return *this;
    }

    // Simple vec operations
    __host__ __device__ __forceinline__ Vec3 operator-() const {
        return Vec3(-e_[0], -e_[1], -e_[2]);
    }
    __host__ __device__ __forceinline__ float length() const { return sqrtf(length_squared()); }
    __host__ __device__ __forceinline__ float length_squared() const {
        return e_[0] * e_[0] + e_[1] * e_[1] + e_[2] * e_[2];
    }
};

// ===== Vector Util Functions =====
__host__ std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
    return os << vec.e_[0] << " " << vec.e_[1] << " " << vec.e_[2];
}

__host__ __device__ __forceinline__ Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[0] + v.e_[0], u.e_[1] + v.e_[1], u.e_[2] + v.e_[2]);
}

__host__ __device__ __forceinline__ Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[0] - v.e_[0], u.e_[1] - v.e_[1], u.e_[2] - v.e_[2]);
}

__host__ __device__ __forceinline__ Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[0] * v.e_[0], u.e_[1] * v.e_[1], u.e_[2] * v.e_[2]);
}

__host__ __device__ __forceinline__ Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t * v.e_[0], t * v.e_[1], t * v.e_[2]);
}

__host__ __device__ __forceinline__ Vec3 operator*(const Vec3& v, float t) {
    return t * v;
}

__host__ __device__ __forceinline__ Vec3 operator/(const Vec3& v, float t) {
    return (1.0f / t) * v;
}

__host__ __device__ __forceinline__ float dot(const Vec3& u, const Vec3& v) {
    return u.e_[0] * v.e_[0] + u.e_[1] * v.e_[1] + u.e_[2] * v.e_[2];
}

__host__ __device__ __forceinline__ Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[1] * v.e_[2] - u.e_[2] * v.e_[1],
                u.e_[2] * v.e_[0] - u.e_[0] * v.e_[2],
                u.e_[0] * v.e_[1] - u.e_[1] * v.e_[0]);
}

__host__ __device__ __forceinline__ Vec3 unit_vector(const Vec3& v) {
    return v / v.length();
}
#endif
