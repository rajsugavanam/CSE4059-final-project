#ifndef SCENE_MANAGER_CUH
#define SCENE_MANAGER_CUH

#include <cuda_runtime.h>

#include <string>

#include "aabb.cuh"
#include "camera.h"
#include "vec3.cuh"
#include "triangle_mesh.cuh"

class SceneManager {
   public:
    // Updated constructor to take a camera reference
    __host__ SceneManager(Camera& camera, int num_objects);
    __host__ ~SceneManager();

    // Prevent copying (RAII objects usually shouldn't be copied)
    SceneManager(const SceneManager&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;

    // Scene setup methods
    __host__ void addAABB(float minx, float miny, float minz, float maxx,
                          float maxy, float maxz, int obj_id);
    __host__ void addTriangleMesh(const std::string& filename, int obj_id = 0);

    // Resource management
    __host__ void allocateResources();
    __host__ void copyToDevice();
    __host__ void copyFromDevice();
    __host__ void freeResources();

    // Save to ppm
    __host__ void saveImage(const char* filename);

    // Main rendering function
    __host__ void renderBox();
    __host__ void renderMesh();

    // Accessors for quick testing
    __host__ void getAABBBounds(Vec3& min_bounds, Vec3& max_bounds) {
        min_bounds = Vec3(h_aabb->h_minx[0], h_aabb[0].h_miny[0],
                            h_aabb[0].h_minz[0]);
        max_bounds = Vec3(h_aabb[0].h_maxx[0], h_aabb[0].h_maxy[0],
                            h_aabb[0].h_maxz[0]);
    }
    __host__ Vec3* getHostImage() { return h_image; }
    __host__ Vec3* getDeviceImage() { return d_image; }
    __host__ int getWidth() const { return width; }
    __host__ int getHeight() const { return height; }
    __host__ int getNumObj() const { return num_objects; }

   private:
    // Store a reference to the camera
    Camera& camera;

    // Image properties
    int width, height;
    Vec3* h_image;  // Host image buffer
    Vec3* d_image;  // Device image buffer

    // Scene objects
    int num_objects;
    AABB* h_aabb;  // Host AABB array
    AABB* d_aabb;  // Device AABB array

    // Triangle mesh data
    int* h_num_triangles;
    int* d_num_triangles;
    TriangleMesh* h_mesh;
    TriangleMesh* d_mesh;


};
#endif  // SCENE_MANAGER_CUH
