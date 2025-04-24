#include <vector>

#include "aabb.cuh"
#include "cuda_helper.h"
#include "ray_color.cuh"
#include "scene_manager.cuh"
#include "timer.h"
#include "triangle3.cuh"
// #include "triangle_mesh.cuh"
#include "vec3.cuh"
#include "obj_reader.cuh"
#include <cassert>

void computeAABBHost(const std::vector<Triangle3>& triangles, Vec3& min_bounds,
                     Vec3& max_bounds);

// bool compareAABB

std::vector<Vec3> hostReduction();
int main() {
    Timer timer;
    CameraParams params;
    Camera camera(params);
    SceneManager scene(camera, 1);

    scene.addTriangleMesh(std::string(PROJECT_ROOT) + "/assets/dk.obj", 0);
    // scene.allocateResources();
    // scene.copyToDevice();
    scene.render();
    scene.copyFromDevice();
    scene.saveImage("mesh_reduction_test.ppm");
    std::cout << "Mesh reduction test completed. Output saved to "
                 "mesh_reduction_test.ppm"
              << std::endl;

    // 
    // Check AABB bounds
    std::vector<Vec3> bounds = hostReduction();
    Vec3 cpu_min_bounds = bounds[0];
    Vec3 cpu_max_bounds = bounds[1];

    // CPU vs GPU AABB comparison
    bool same_aabb = false;
    float epsilon = 1e-5f;
    
    // Get AABB bounds from GPU (from scene manager)
    Vec3 gpu_min_bounds, gpu_max_bounds;
    scene.getAABBBounds(gpu_min_bounds, gpu_max_bounds);
    
    // Compare CPU and GPU results with epsilon tolerance
    if (std::abs(gpu_min_bounds.x() - cpu_min_bounds.x()) < epsilon && 
        std::abs(gpu_min_bounds.y() - cpu_min_bounds.y()) < epsilon && 
        std::abs(gpu_min_bounds.z() - cpu_min_bounds.z()) < epsilon &&
        std::abs(gpu_max_bounds.x() - cpu_max_bounds.x()) < epsilon && 
        std::abs(gpu_max_bounds.y() - cpu_max_bounds.y()) < epsilon && 
        std::abs(gpu_max_bounds.z() - cpu_max_bounds.z()) < epsilon) {
        same_aabb = true;
    }

    // Print GPU AABB results
    std::cout << "GPU AABB min: " << gpu_min_bounds.x() << ", "
              << gpu_min_bounds.y() << ", " << gpu_min_bounds.z() << "\n";
    std::cout << "GPU AABB max: " << gpu_max_bounds.x() << ", "
              << gpu_max_bounds.y() << ", " << gpu_max_bounds.z() << "\n";

    // Print CPU AABB results
    std::cout << "CPU AABB min: " << cpu_min_bounds.x() << ", "
              << cpu_min_bounds.y() << ", " << cpu_min_bounds.z() << "\n";
    std::cout << "CPU AABB max: " << cpu_max_bounds.x() << ", "
              << cpu_max_bounds.y() << ", " << cpu_max_bounds.z() << "\n";

    if (!same_aabb) {
        std::cerr << "AABB bounds do not match!" << std::endl;
        return 1;
    }
    std::cout << "AABB bounds match!" << std::endl;
    return 0;
}

std::vector<Vec3> hostReduction() {
    Timer timer;
    ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/dk.obj");
    reader.readModel();
    Model model = reader.parsedModel;
    assert(!model.modelTriangles.empty());
    timer.start("CPU AABB computation");
    Vec3 cpu_min_bounds, cpu_max_bounds;
    computeAABBHost(model.modelTriangles, cpu_min_bounds, cpu_max_bounds);
    timer.stop();

    return {cpu_min_bounds, cpu_max_bounds};
}

// Add a CPU function to compute AABB bounds for comparison
void computeAABBHost(const std::vector<Triangle3>& triangles, Vec3& min_bounds,
                     Vec3& max_bounds) {
    if (triangles.empty()) return;

    // Initialize bounds with first vertex of first triangle
    min_bounds = triangles[0].vertex0();
    max_bounds = triangles[0].vertex0();

    // Check all vertices of all triangles
    for (const auto& triangle : triangles) {
        // Check vertex 0
        min_bounds = Vec3(fminf(min_bounds.x(), triangle.vertex0().x()),
                          fminf(min_bounds.y(), triangle.vertex0().y()),
                          fminf(min_bounds.z(), triangle.vertex0().z()));
        max_bounds = Vec3(fmaxf(max_bounds.x(), triangle.vertex0().x()),
                          fmaxf(max_bounds.y(), triangle.vertex0().y()),
                          fmaxf(max_bounds.z(), triangle.vertex0().z()));

        // Check vertex 1
        min_bounds = Vec3(fminf(min_bounds.x(), triangle.vertex1().x()),
                          fminf(min_bounds.y(), triangle.vertex1().y()),
                          fminf(min_bounds.z(), triangle.vertex1().z()));
        max_bounds = Vec3(fmaxf(max_bounds.x(), triangle.vertex1().x()),
                          fmaxf(max_bounds.y(), triangle.vertex1().y()),
                          fmaxf(max_bounds.z(), triangle.vertex1().z()));

        // Check vertex 2
        min_bounds = Vec3(fminf(min_bounds.x(), triangle.vertex2().x()),
                          fminf(min_bounds.y(), triangle.vertex2().y()),
                          fminf(min_bounds.z(), triangle.vertex2().z()));
        max_bounds = Vec3(fmaxf(max_bounds.x(), triangle.vertex2().x()),
                          fmaxf(max_bounds.y(), triangle.vertex2().y()),
                          fmaxf(max_bounds.z(), triangle.vertex2().z()));
    }
}