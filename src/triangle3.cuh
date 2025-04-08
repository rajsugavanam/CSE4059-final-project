#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "vec3.cuh"
class Triangle3 {
   public:
    __host__ __device__ Triangle3()
        : vertex0_{Vec3()}, vertex1_{Vec3()}, vertex2_{Vec3()} {}
    __host__ __device__ Triangle3(Vec3 vertex0, Vec3 vertex1, Vec3 vertex2)
        : vertex0_{vertex0}, vertex1_{vertex1}, vertex2_{vertex2} {}

    __host__ __device__ const Vec3 vertex0() const { return vertex0_; }
    __host__ __device__ const Vec3 vertex1() const { return vertex1_; }
    __host__ __device__ const Vec3 vertex2() const { return vertex2_; }
    __host__ __device__ const Vec3 edge0() const { return vertex1_ - vertex0_; }

    __host__ __device__ const Vec3 edge1() const { return vertex2_ - vertex0_; }

    __host__ __device__ const Vec3 edge2() const { return vertex2_ - vertex1_; }

   private:
    Vec3 vertex0_, vertex1_, vertex2_;
};

#endif
