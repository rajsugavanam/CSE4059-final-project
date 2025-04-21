#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "vec3.cuh"

class Triangle3 {
   public:
    Vec3 vertex0_, vertex1_, vertex2_;
    Vec3 normal0_, normal1_, normal2_;
    
    // Constructors
    __host__ __device__ Triangle3()
        : vertex0_{Vec3()}, vertex1_{Vec3()}, vertex2_{Vec3()},
          normal0_{Vec3()}, normal1_{Vec3()}, normal2_{Vec3()} {}
          
    __host__ __device__ Triangle3(Vec3 vertex0, Vec3 vertex1, Vec3 vertex2)
        : vertex0_{vertex0}, vertex1_{vertex1}, vertex2_{vertex2},
          normal0_{Vec3()}, normal1_{Vec3()}, normal2_{Vec3()} {}
          
    __host__ __device__ Triangle3(Vec3 vertex0, Vec3 vertex1, Vec3 vertex2,
                                 Vec3 normal0, Vec3 normal1, Vec3 normal2)
        : vertex0_{vertex0}, vertex1_{vertex1}, vertex2_{vertex2},
          normal0_{normal0}, normal1_{normal1}, normal2_{normal2} {}

    // Vertex accessors
    __host__ __device__ __forceinline__ const Vec3 vertex0() const { return vertex0_; }
    __host__ __device__ __forceinline__ const Vec3 vertex1() const { return vertex1_; }
    __host__ __device__ __forceinline__ const Vec3 vertex2() const { return vertex2_; }
    
    // Normal accessors
    __host__ __device__ __forceinline__ const Vec3 normal0() const { return normal0_; }
    __host__ __device__ __forceinline__ const Vec3 normal1() const { return normal1_; }
    __host__ __device__ __forceinline__ const Vec3 normal2() const { return normal2_; }
    
    // Edge calculations
    __host__ __device__ __forceinline__ const Vec3 edge0() const { return vertex1_ - vertex0_; }
    __host__ __device__ __forceinline__ const Vec3 edge1() const { return vertex2_ - vertex0_; }
    __host__ __device__ __forceinline__ const Vec3 edge2() const { return vertex2_ - vertex1_; }
};

#endif
