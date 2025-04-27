#ifndef SCENE_MANAGER_CUH
#define SCENE_MANAGER_CUH

#include "aabb.cuh"
#include "camera.h"
#include "material.cuh"
#include "triangle_mesh.cuh"
// #include "vec3.h"

class SceneManager {
  public:
    // Constructor
    __host__ SceneManager(Camera& camera, int num_objects);

    // Destructor
    __host__ ~SceneManager();

    // Add objects to scene
    __host__ void addAABB(float minx, float miny, float minz, float maxx,
                          float maxy, float maxz, int obj_id);
    __host__ void addTriangleMesh(const std::string& filename, int obj_id);
    __host__ void addTriangleMeshColor(const std::string& filename, float3 albedo,
                                  int obj_id);
    __host__ void addTriangleMeshSpectrum(const std::string& filename,
                                   int spectral_reflectance_id,
                                   int spectral_emission_id,
                                   bool is_emissive,
                                   int obj_id);

    // Initialize spectral data
    __host__ void initializeSpectra();

    // Render the scene
    __host__ void renderBox();
    __host__ void renderMesh();
    __host__ void renderSpectralMesh(int samples_per_pixel = 16); // Added parameter with default value
    
    // Copy data between host and device
    __host__ void copyToDevice();
    __host__ void copyFromDevice();

    // Save the rendered image
    __host__ void saveImage(const char* filename);
    
    // Get AABB bounds for all objects
    __host__ void getAABBBounds(Vec3 & min_bounds,
                                Vec3& max_bounds);

  private:  
    // Allocation and deallocation
    __host__ void allocateResources();
    __host__ void freeResources();

  public:
    // Scene properties
    Camera& camera;
    int width;
    int height;
    int num_objects;
    Material materials[256];

    // Host data
    Vec3* h_image;
    AABB* h_aabb;
    int* h_num_triangles;
    TriangleMesh* h_mesh;

    // Device data
    Vec3* d_image;
    AABB* d_aabb;
    int* d_num_triangles;
    TriangleMesh* d_mesh;
};

#endif  // SCENE_MANAGER_CUH
