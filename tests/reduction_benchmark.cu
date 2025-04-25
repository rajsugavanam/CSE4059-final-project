#include <iostream>

#include "aabb.cuh"
#include "timer.h"
#include "triangle_mesh.cuh"
#include "reduction.cuh"

int main() {
    Timer timer;
    std::string obj_file = std::string(PROJECT_ROOT) + "/assets/dk.obj";
    
    // Load mesh
    timer.start("Loading mesh");
    TriangleMesh* mesh = TriangleMesh::loadFromOBJ(obj_file);
    if (!mesh) {
        std::cerr << "Failed to load mesh" << std::endl;
        return 1;
    }
    timer.stop();
    
    std::cout << "Loaded mesh with " << mesh->numTriangles() << " triangles" << std::endl;
    
    // Benchmark CPU reduction
    timer.start("CPU AABB reduction");
    AABB cpuAABB(1);
    cpuAABB.mallocAABB();
    mesh->meshMemcpyHtD();
    mesh->computeAABB(&cpuAABB, 0);
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
