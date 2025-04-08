#ifndef VEC3_CUH
#define VEC3_CUH

#include <iostream>

// maybe change to template for double precision
class Vec3 {
  public:
    float e_[3];

    // Constructors
    __host__ __device__ Vec3();
    __host__ __device__ Vec3(float e0, float e1, float e2);
    // __host__ __device__ Vec3(float e0, float e1, float e2) : e_{e0, e1, e2} {}
    // Accessors
    __host__ __device__ float x() const;
    __host__ __device__ float y() const;
    __host__ __device__ float z() const;

    // member functions
    __host__ __device__ Vec3 operator-() const;
    __host__ __device__ float operator[](int idx) const;
    __host__ __device__ float& operator[](int idx);

    __host__ __device__ Vec3& operator+=(const Vec3& vec3);
    __host__ __device__ Vec3& operator*=(float scalar);
    __host__ __device__ Vec3& operator/=(float scalar);

    __host__ __device__ float length() const;
    __host__ __device__ float length_squared() const;
};

// ===== Vector Util Functions =====
__host__ std::ostream& operator<<(std::ostream& os, const Vec3& vec);

__host__ __device__ Vec3 operator+(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 operator-(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 operator*(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 operator*(float t, const Vec3& v);

__host__ __device__ Vec3 operator*(const Vec3& v, float t);

__host__ __device__ Vec3 operator/(const Vec3& v, float t);

__host__ __device__ float dot(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 cross(const Vec3& u, const Vec3& v);

__host__ __device__ Vec3 unit_vector(const Vec3& v);
#endif