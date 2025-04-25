#include "triangle_mesh.cuh"
#include "vec3.cuh"
#include "cuda_math.h"

int main() {
    bool test = 0;
    Vec3 v1(1.0f, 2.0f, 3.0f);
    Vec3 v2(4.0f, 5.0f, 6.0f);

    // Replace float arrays with float3 types
    float3 vec1 = make_float3(1.0f, 2.0f, 3.0f);
    float3 vec2 = make_float3(4.0f, 5.0f, 6.0f);

    float scalar = 2.0f;

    // Test vector addition
    Vec3 result = v1 + v2;
    float3 v_result = vec1 + vec2; // Using operator overloading

    if (result.x() != v_result.x || result.y() != v_result.y ||
        result.z() != v_result.z) {
        std::cout << "Vector addition test failed!" << std::endl;
        test = 1;
    }
    std::cout << "Vector addition result: " << result << std::endl;
    std::cout << "CUDA vector addition result: " << v_result.x << " "
              << v_result.y << " " << v_result.z << std::endl;

    // Test vector subtraction
    Vec3 result_sub = v1 - v2;
    float3 v_result_sub = vec1 - vec2; // Using operator overloading
    
    if (result_sub.x() != v_result_sub.x ||
        result_sub.y() != v_result_sub.y ||
        result_sub.z() != v_result_sub.z) {
        std::cout << "Vector subtraction test failed!" << std::endl;
        test = 1;
    }
    std::cout << "Vector subtraction result: " << result_sub << std::endl;
    std::cout << "CUDA vector subtraction result: " << v_result_sub.x << " "
              << v_result_sub.y << " " << v_result_sub.z << std::endl;

    // Test dot product
    float cuda_dot_result = dot(vec1, vec2);
    float expected_dot = dot(v1, v2);
    if (cuda_dot_result != expected_dot) {
        std::cout << "Dot product test failed!" << std::endl;
        test = 1;
    }
    std::cout << "Dot product result: " << cuda_dot_result << std::endl;

    // Test scalar multiplication
    result = v1 * scalar;
    float3 v_result_mult = vec1 * scalar; // Using operator overloading
    
    if (result.x() != v_result_mult.x || result.y() != v_result_mult.y ||
        result.z() != v_result_mult.z) {
        std::cout << "Scalar multiplication test failed!" << std::endl;
        test = 1;
    }
    std::cout << "Scalar multiplication result: " << result << std::endl;
    std::cout << "CUDA scalar multiplication result: " << v_result_mult.x << " "
              << v_result_mult.y << " " << v_result_mult.z << std::endl;

    // Test cross product
    result = cross(v1, v2);
    float3 v_result_cross = cross(vec1, vec2);
    
    if (result.x() != v_result_cross.x || result.y() != v_result_cross.y ||
        result.z() != v_result_cross.z) {
        std::cout << "Cross product test failed!" << std::endl;
        test = 1;
    }
    std::cout << "Cross product result: " << result << std::endl;
    std::cout << "CUDA cross product result: " << v_result_cross.x << " "
              << v_result_cross.y << " " << v_result_cross.z << std::endl;

    // Normalize test
    Vec3 normalized = unit_vector(v1);
    float3 v_normalized = normalize(vec1);
    
    if (normalized.x() != v_normalized.x || normalized.y() != v_normalized.y ||
        normalized.z() != v_normalized.z) {
        std::cout << "Normalization test failed!" << std::endl;
        test = 1;
    }
    std::cout << "Normalization result: " << normalized << std::endl;
    std::cout << "CUDA normalization result: " << v_normalized.x << " "
              << v_normalized.y << " " << v_normalized.z << std::endl;

    return test;
}