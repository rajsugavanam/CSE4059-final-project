#include <iostream>

#include "aabb.cuh"
#include "timer.h"
#include "triangle_mesh.cuh"
#include "reduction.cuh"
#include "obj_reader.cuh"

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
int main() {
    Timer timer;
    std::string obj_file = std::string(PROJECT_ROOT) + "/assets/dragon.obj";
    
    // Load mesh
    timer.start("Loading mesh");
    TriangleMesh* mesh = TriangleMesh::loadFromOBJ(obj_file);
    if (!mesh) {
        std::cerr << "Failed to load mesh" << std::endl;
        return 1;
    }
    timer.stop();
    
    std::cout << "Loaded mesh with " << mesh->numTriangles() << " triangles" << std::endl;
    mesh->meshMemcpyHtD();
    // Benchmark CPU reduction
    
    // Benchmark CPU reduction
    ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/dragon.obj");
    reader.readModel();
    Model model = reader.parsedModel;
    Vec3 cpu_min_bounds = Vec3(INFINITY, INFINITY, INFINITY);
    Vec3 cpu_max_bounds = Vec3(-INFINITY, -INFINITY, -INFINITY);
    AABB cpuAABB(1);
    timer.start("CPU AABB reduction");
    computeAABBHost(model.modelTriangles, cpu_min_bounds, cpu_max_bounds);
    cpuAABB.h_minx[0] = cpu_min_bounds.x();
    cpuAABB.h_miny[0] = cpu_min_bounds.y();
    cpuAABB.h_minz[0] = cpu_min_bounds.z();
    cpuAABB.h_maxx[0] = cpu_max_bounds.x();
    cpuAABB.h_maxy[0] = cpu_max_bounds.y();
    cpuAABB.h_maxz[0] = cpu_max_bounds.z();
    timer.stop();
   
    
    // Benchmark standard GPU reduction
    timer.start("Standard GPU reduction");
    float min_bounds[3] = {INFINITY, INFINITY, INFINITY};
    float max_bounds[3] = {-INFINITY, -INFINITY, -INFINITY};
    computeMeshReduction(mesh, 0, min_bounds, max_bounds);
    timer.stop();
    
    // Benchmark streamed GPU reduction
    timer.start("Streamed GPU reduction");
    float min_bounds_streamed[3] = {INFINITY, INFINITY, INFINITY};
    float max_bounds_streamed[3] = {-INFINITY, -INFINITY, -INFINITY};
    computeMeshReductionStreams(mesh, 0, min_bounds_streamed, max_bounds_streamed);
    timer.stop();
    
    // Output and compare results
    std::cout << "\nResults comparison:" << std::endl;
    std::cout << "CPU min bounds: (" << cpuAABB.h_minx[0] << ", " << cpuAABB.h_miny[0] << ", " << cpuAABB.h_minz[0] << ")" << std::endl;
    std::cout << "CPU max bounds: (" << cpuAABB.h_maxx[0] << ", " << cpuAABB.h_maxy[0] << ", " << cpuAABB.h_maxz[0] << ")" << std::endl;
    
    std::cout << "Standard GPU min: (" << min_bounds[0] << ", " << min_bounds[1] << ", " << min_bounds[2] << ")" << std::endl;
    std::cout << "Standard GPU max: (" << max_bounds[0] << ", " << max_bounds[1] << ", " << max_bounds[2] << ")" << std::endl;
    
    std::cout << "Streamed GPU min: (" << min_bounds_streamed[0] << ", " << min_bounds_streamed[1] << ", " << min_bounds_streamed[2] << ")" << std::endl;
    std::cout << "Streamed GPU max: (" << max_bounds_streamed[0] << ", " << max_bounds_streamed[1] << ", " << max_bounds_streamed[2] << ")" << std::endl;
    
    // Cleanup
    delete mesh;

    if (cpuAABB.h_minx[0] != min_bounds[0] || cpuAABB.h_miny[0] != min_bounds[1] || cpuAABB.h_minz[0] != min_bounds[2]) {
        std::cerr << "Mismatch in min bounds!" << std::endl;
        return 1;
    }
    if (cpuAABB.h_maxx[0] != max_bounds[0] || cpuAABB.h_maxy[0] != max_bounds[1] || cpuAABB.h_maxz[0] != max_bounds[2]) {
        std::cerr << "Mismatch in max bounds!" << std::endl;
        return 1;
    }
    if (cpuAABB.h_minx[0] != min_bounds_streamed[0] || cpuAABB.h_miny[0] != min_bounds_streamed[1] || cpuAABB.h_minz[0] != min_bounds_streamed[2]) {
        std::cerr << "Mismatch in streamed min bounds!" << std::endl;
        return 1;
    }
    if (cpuAABB.h_maxx[0] != max_bounds_streamed[0] || cpuAABB.h_maxy[0] != max_bounds_streamed[1] || cpuAABB.h_maxz[0] != max_bounds_streamed[2]) {
        std::cerr << "Mismatch in streamed max bounds!" << std::endl;
        return 1;
    }
    std::cout << "All bounds match!" << std::endl;
    return 0;
}
