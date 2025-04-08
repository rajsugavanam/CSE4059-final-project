#include "vec3.cuh"

// Implementation of Vec3 member functions
__host__ __device__ Vec3::Vec3() : e_{0.0f, 0.0f, 0.0f} {}
__host__ __device__ Vec3::Vec3(float e0, float e1, float e2) : e_{e0, e1, e2} {}

__host__ __device__ float Vec3::x() const { return e_[0]; }
__host__ __device__ float Vec3::y() const { return e_[1]; }
__host__ __device__ float Vec3::z() const { return e_[2]; }

__host__ __device__ Vec3 Vec3::operator-() const {
    return Vec3(-e_[0], -e_[1], -e_[2]);
}

__host__ __device__ float Vec3::operator[](int i) const { return e_[i]; }

__host__ __device__ float& Vec3::operator[](int i) { return e_[i]; }

__host__ __device__ Vec3& Vec3::operator+=(const Vec3& vec) {
    e_[0] += vec.e_[0];
    e_[1] += vec.e_[1];
    e_[2] += vec.e_[2];
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(float scalar) {
    e_[0] *= scalar;
    e_[1] *= scalar;
    e_[2] *= scalar;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator/=(float scalar) {
    e_[0] *= 1.0f / scalar;
    e_[1] *= 1.0f / scalar;
    e_[2] *= 1.0f / scalar;
    return *this;
}

__host__ __device__ float Vec3::length() const {
    return sqrtf(length_squared());
}

__host__ __device__ float Vec3::length_squared() const {
    return e_[0] * e_[0] + e_[1] * e_[1] + e_[2] * e_[2];
}

// Vector utility functions
__host__ std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
    return os << vec.e_[0] << " " << vec.e_[1] << " " << vec.e_[2];
}

__host__ __device__ Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[0] + v.e_[0], u.e_[1] + v.e_[1], u.e_[2] + v.e_[2]);
}

__host__ __device__ Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[0] - v.e_[0], u.e_[1] - v.e_[1], u.e_[2] - v.e_[2]);
}

__host__ __device__ Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[0] * v.e_[0], u.e_[1] * v.e_[1], u.e_[2] * v.e_[2]);
}

__host__ __device__ Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t * v.e_[0], t * v.e_[1], t * v.e_[2]);
}

__host__ __device__ Vec3 operator*(const Vec3& v, float t) {
    return Vec3(t * v.e_[0], t * v.e_[1], t * v.e_[2]);
}

__host__ __device__ Vec3 operator/(const Vec3& v, float t) {
    return Vec3(v.e_[0] / t, v.e_[1] / t, v.e_[2] / t);
}

__host__ __device__ float dot(const Vec3& u, const Vec3& v) {
    return u.e_[0] * v.e_[0] + u.e_[1] * v.e_[1] + u.e_[2] * v.e_[2];
}

__host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.e_[1] * v.e_[2] - u.e_[2] * v.e_[1],
                u.e_[2] * v.e_[0] - u.e_[0] * v.e_[2],
                u.e_[0] * v.e_[1] - u.e_[1] * v.e_[0]);
}

__host__ __device__ Vec3 unit_vector(const Vec3& v) {
    return v / v.length();
}