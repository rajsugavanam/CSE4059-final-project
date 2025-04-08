#ifndef RAY_CUH
#define RAY_CUH

#include "vec3.cuh"

class Ray {
   public:
    __host__ __device__ Ray();

    __host__ __device__ Ray(const Vec3& origin, const Vec3& direction)
        : origin_(origin), direction_(direction) {}

    __host__ __device__ const Vec3& origin() const { return origin_; }
    __host__ __device__ const Vec3& direction() const { return direction_; }

    __host__ __device__ Vec3 at(double scal) const {
        return origin_ + scal * direction_;
    }

   private:
    Vec3 origin_;
    Vec3 direction_;
};
#endif
