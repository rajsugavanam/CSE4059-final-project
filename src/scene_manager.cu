// #include <fstream>
#include <iostream>

#include "camera.h"
#include "cuda_helper.h"
#include "image_writer.h"
#include "model.cuh"
#include "obj_reader.cuh"
#include "ray_color.cuh"
#include "scene_manager.cuh"
#include "timer.h"
#include "triangle_mesh.cuh"

// Möller–Trumbore intersection algorithm for TriangleMesh
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool rayIntersectsTriangle(const Ray& ray, const TriangleMesh* mesh,
                                      int tri_id, int tri_idx,
                                      Vec3& intersection, float& u, float& v,
                                      float& t) {
    const float epsilon = 1.19209e-07f;  // float32 machine epsilon ish

    // Store vertex coordinates
    float v0x = mesh[tri_id].d_v0x[tri_idx];
    float v0y = mesh[tri_id].d_v0y[tri_idx];
    float v0z = mesh[tri_id].d_v0z[tri_idx];
    float v1x = mesh[tri_id].d_v1x[tri_idx];
    float v1y = mesh[tri_id].d_v1y[tri_idx];
    float v1z = mesh[tri_id].d_v1z[tri_idx];
    float v2x = mesh[tri_id].d_v2x[tri_idx];
    float v2y = mesh[tri_id].d_v2y[tri_idx];
    float v2z = mesh[tri_id].d_v2z[tri_idx];

    // Get ray components
    float ray_ox = ray.origin().x();
    float ray_oy = ray.origin().y();
    float ray_oz = ray.origin().z();
    float ray_dx = ray.direction().x();
    float ray_dy = ray.direction().y();
    float ray_dz = ray.direction().z();

    // Calculate edges as float components
    float edge1x = v1x - v0x;
    float edge1y = v1y - v0y;
    float edge1z = v1z - v0z;
    float edge2x = v2x - v0x;
    float edge2y = v2y - v0y;
    float edge2z = v2z - v0z;
    
    // Calculate ray_cross_e2 components (cross product)
    float rxe2x = ray_dy * edge2z - ray_dz * edge2y;
    float rxe2y = ray_dz * edge2x - ray_dx * edge2z;
    float rxe2z = ray_dx * edge2y - ray_dy * edge2x;
    
    // Calculate determinant (dot product)
    float determinant = edge1x * rxe2x + edge1y * rxe2y + edge1z * rxe2z;

    // Parallel to triangle
    if (fabsf(determinant) < epsilon) {
        return false;
    }

    float inv_determinant = 1.0f / determinant;
    
    // Calculate s vector components directly
    float sx = ray_ox - v0x;
    float sy = ray_oy - v0y;
    float sz = ray_oz - v0z;
    
    // Calculate dot(s, ray_cross_e2) directly
    float dot_s_rce2 = sx * rxe2x + sy * rxe2y + sz * rxe2z;
    u = inv_determinant * dot_s_rce2;

    // Out of bounds
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    // Calculate s_cross_e1 components directly
    float scrossex = sy * edge1z - sz * edge1y;
    float scrossey = sz * edge1x - sx * edge1z;
    float scrossez = sx * edge1y - sy * edge1x;
    
    // Calculate dot(ray.direction(), s_cross_e1) directly
    float dot_dir_sce1 = ray_dx * scrossex + 
                          ray_dy * scrossey + 
                          ray_dz * scrossez;
    v = inv_determinant * dot_dir_sce1;

    // Out of bounds
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    // Compute t to find intersection point - dot(edge2, s_cross_e1)
    float dot_e2_sce1 = edge2x * scrossex + edge2y * scrossey + edge2z * scrossez;
    t = inv_determinant * dot_e2_sce1;

    if (t > epsilon) {
        // Calculate intersection point directly
        intersection = Vec3(
            ray_ox + ray_dx * t,
            ray_oy + ray_dy * t,
            ray_oz + ray_dz * t
        );
        return true;
    }
    return false;
}

// Color ray based on hit triangle
// FIRST CHECK AABB, THEN CHECK TRIANGLE
__device__ Vec3 colorRayTriangle(const Ray& ray, const AABB* boxes, const TriangleMesh* mesh,
                                  int num_objects, const int* num_triangles) {
    float closest_t = INFINITY;
    int hit_obj_id = -1;
    int hit_tri_idx = -1;
    Vec3 hit_point;
    float hit_u, hit_v;

    // Check intersection with any object in the array
    for (int obj_id = 0; obj_id < num_objects; obj_id++) {
        // First check AABB hit to avoid unnecessary triangle checks
        if (boxes->hitAABB(ray, obj_id)) {
            // Now check all triangles in this mesh
            for (int tri_idx = 0; tri_idx < num_triangles[obj_id]; tri_idx++) {
                Vec3 intersection;
                float u, v, t;
                
                // Check intersection with this triangle
                if (rayIntersectsTriangle(ray, mesh, obj_id, tri_idx, 
                                        intersection, u, v, t)) {
                    // Keep track of the closest intersection
                    if (t < closest_t) {
                        closest_t = t;
                        hit_obj_id = obj_id;
                        hit_tri_idx = tri_idx;
                        hit_point = intersection;
                        hit_u = u;
                        hit_v = v;
                    }
                }
            }
        }
    }
    
    // If we hit a triangle, return its color
    if (hit_obj_id >= 0) {
        // Get the barycentric coordinates for color interpolation
        float w = 1.0f - hit_u - hit_v;  // third barycentric coordinate

        Vec3 normal = Vec3(w * mesh[hit_obj_id].d_n0x[hit_tri_idx] +
                           hit_u * mesh[hit_obj_id].d_n1x[hit_tri_idx] +
                           hit_v * mesh[hit_obj_id].d_n2x[hit_tri_idx],
                           w * mesh[hit_obj_id].d_n0y[hit_tri_idx] +
                           hit_u * mesh[hit_obj_id].d_n1y[hit_tri_idx] +
                           hit_v * mesh[hit_obj_id].d_n2y[hit_tri_idx],
                           w * mesh[hit_obj_id].d_n0z[hit_tri_idx] +
                           hit_u * mesh[hit_obj_id].d_n1z[hit_tri_idx] +
                           hit_v * mesh[hit_obj_id].d_n2z[hit_tri_idx]);

        return normalMap(normal);  // Color based on normal
        // Simple coloring based on object ID and triangle index
        // return threeColor(hit_obj_id);
    }

    // No hit, return sky background
    return skyBg(ray);
}

__device__ Vec3 colorRayBox(const Ray& ray, const AABB* boxes,
                            int num_objects) {
    // Check intersection with any box in the array
    for (int i = 0; i < num_objects; i++) {
        if (boxes->hitAABB(ray, i)) {
            // 
            switch (i % 3) {
                case 0:
                    return Vec3(1.0f, 0.0f, 0.0f);  // red
                case 1:
                    return Vec3(0.0f, 1.0f, 0.0f);  // green
                default:
                    return Vec3(0.0f, 0.0f, 1.0f);  // white
            }
        }
    }
    // Background color (gradient from blue to white)
    return skyBg(ray);
}

// Modified kernel to use an array of AABB objects
__global__ void renderBoxKernel(Vec3* image_buffer,
                                CUDACameraParams camera_params, AABB* boxes,
                                int num_objects) {
    int width = camera_params.pixel_width;
    int height = camera_params.pixel_height;
    Vec3 pixel00_loc = camera_params.pixel00_loc;
    Vec3 delta_u = camera_params.pixel_delta_u;
    Vec3 delta_v = camera_params.pixel_delta_v;
    Vec3 camera_origin = camera_params.center;

    // Calculate pixel coordinates
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        const int pixel_idx = row * width + col;

        // ray params
        const Vec3 pixel_center =
            pixel00_loc + (col * delta_u) + (row * delta_v);
        const Vec3 ray_direction = pixel_center - camera_origin;

        Ray ray(camera_origin, ray_direction);
        image_buffer[pixel_idx] = colorRayBox(ray, boxes, num_objects);
    }
}

// AABBB TRIANGLE MESH RENDERING
__global__ void renderMeshKernel(Vec3* image_buffer, AABB* boxes, TriangleMesh* meshes,
                                 const int num_objects, const int* __restrict__ num_triangles,
                                 const CUDACameraParams camera_params) {
    
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < camera_params.pixel_width && row < camera_params.pixel_height) {
        const int pixel_idx = row * camera_params.pixel_width + col;

        // ray params
        const Vec3 pixel_center = camera_params.pixel00_loc +
                                  (col * camera_params.pixel_delta_u) +
                                  (row * camera_params.pixel_delta_v);
        const Vec3 ray_direction = pixel_center - camera_params.center;

        Ray ray(camera_params.center, ray_direction);
        image_buffer[pixel_idx] = colorRayTriangle(ray, boxes, meshes, num_objects, num_triangles);
    }
}

__host__ SceneManager::SceneManager(Camera& camera, int num_objects)
    : camera(camera),
      width(camera.pixelWidth()),
      height(camera.pixelHeight()),
      num_objects(num_objects),
      h_image(nullptr),
      d_image(nullptr),
      h_aabb(nullptr),
      d_aabb(nullptr),
      h_num_triangles(nullptr),
      d_num_triangles(nullptr),
      h_mesh(nullptr),
      d_mesh(nullptr) {
    allocateResources();

    std::cout << "Scene dimensions: " << width << "x" << height << std::endl;
}

// Destructor - automatically free resources (RAII)
__host__ SceneManager::~SceneManager() {
    std::cout << "Freeing resources..." << std::endl;
    freeResources();
}

// Add an AABB to the scene
__host__ void SceneManager::addAABB(float minx, float miny, float minz,
                                    float maxx, float maxy, float maxz,
                                    int obj_id) {
    // Add the AABB to the host array given the object ID
    h_aabb->h_minx[obj_id] = minx;
    h_aabb->h_miny[obj_id] = miny;
    h_aabb->h_minz[obj_id] = minz;
    h_aabb->h_maxx[obj_id] = maxx;
    h_aabb->h_maxy[obj_id] = maxy;
    h_aabb->h_maxz[obj_id] = maxz;
    h_aabb->num_obj = num_objects;

    std::cout << "Added AABB box: (" << minx << ", " << miny << ", " << minz
              << ") to (" << maxx << ", " << maxy << ", " << maxz << ")"
              << " :: Number of Objects: " << obj_id + 1 << std::endl;
}

// Add triangle mesh from OBJ file
__host__ void SceneManager::addTriangleMesh(const std::string& filename,
                                            int obj_id) {
    std::cout << "Loading triangle mesh from: " << filename
              << " for object ID: " << obj_id << std::endl;

    // Check if object ID is valid
    if (obj_id < 0 || obj_id >= num_objects) {
        std::cerr << "Invalid object ID: " << obj_id << std::endl;
        return;
    }

    // Use ObjReader directly to load the mesh
    ObjReader reader(filename);

    // Make sure the file is read successfully
    reader.readModel();

    // Get the triangles from the model
    const std::vector<Triangle3>& triangles = reader.parsedModel.modelTriangles;
    h_num_triangles[obj_id] = triangles.size();

    if (h_num_triangles[obj_id] == 0) {
        std::cerr << "Error: No triangles found in OBJ file: " << filename
                  << std::endl;
        return;
    }

    std::cout << "Successfully loaded " << h_num_triangles[obj_id] << " triangles from "
              << filename << std::endl;

    // Clear previous mesh data at this index 
    h_mesh[obj_id].~TriangleMesh();

    // Initialize a new mesh at this index
    new (&h_mesh[obj_id]) TriangleMesh(h_num_triangles[obj_id]);

    // Copy triangle data to mes
    for (int i = 0; i < h_num_triangles[obj_id]; i++) {
        // Copy vertices
        h_mesh[obj_id].h_v0x[i] = triangles[i].vertex0().x();
        h_mesh[obj_id].h_v0y[i] = triangles[i].vertex0().y();
        h_mesh[obj_id].h_v0z[i] = triangles[i].vertex0().z();
        h_mesh[obj_id].h_v1x[i] = triangles[i].vertex1().x();
        h_mesh[obj_id].h_v1y[i] = triangles[i].vertex1().y();
        h_mesh[obj_id].h_v1z[i] = triangles[i].vertex1().z();
        h_mesh[obj_id].h_v2x[i] = triangles[i].vertex2().x();
        h_mesh[obj_id].h_v2y[i] = triangles[i].vertex2().y();
        h_mesh[obj_id].h_v2z[i] = triangles[i].vertex2().z();
        // Copy normals
        h_mesh[obj_id].h_n0x[i] = triangles[i].normal0().x();
        h_mesh[obj_id].h_n0y[i] = triangles[i].normal0().y();
        h_mesh[obj_id].h_n0z[i] = triangles[i].normal0().z();
        h_mesh[obj_id].h_n1x[i] = triangles[i].normal1().x();
        h_mesh[obj_id].h_n1y[i] = triangles[i].normal1().y();
        h_mesh[obj_id].h_n1z[i] = triangles[i].normal1().z();
        h_mesh[obj_id].h_n2x[i] = triangles[i].normal2().x();
        h_mesh[obj_id].h_n2y[i] = triangles[i].normal2().y();
        h_mesh[obj_id].h_n2z[i] = triangles[i].normal2().z();
    }

    // Copy mesh data to device memory
    h_mesh[obj_id].meshMemcpyHtD();

    if (h_aabb != nullptr) {
        h_mesh->computeAABB(h_aabb, obj_id);
    }

    // Make sure the data is copied to the device
    copyToDevice();

    std::cout << "Mesh loaded with " << h_mesh[obj_id].numTriangles()
              << " triangles and added to scene as object " << obj_id
              << std::endl;
}

// Allocate GPU resources
__host__ void SceneManager::allocateResources() {
    std::cout << "Allocating GPU resources" << std::endl;
    // Allocate host image buffer
    h_image = new Vec3[width * height];

    // Allocate device image buffer
    CUDA_CHECK(cudaMalloc(&d_image, width * height * sizeof(Vec3)));

    // Allocate device AABB array
    if (num_objects > 0) {
        // Allocate a single AABB object that will handle multiple AABBs
        h_aabb = new AABB[1];

        // Initialize the AABB with the constructor that handles memory
        // allocation
        *h_aabb = AABB(num_objects);
        h_aabb->mallocAABB();

        // Allocate GPU memory for AABB data
        h_aabb->cudaMallocAABB();

        // Allocate device memory for the AABB object
        CUDA_CHECK(cudaMalloc(&d_aabb, sizeof(AABB)));

        // Allocate contiguous array of TriangleMesh objects (not pointers)
        h_mesh = new TriangleMesh[num_objects];

        // Allocate device memory for the array of mesh objects
        CUDA_CHECK(cudaMalloc(&d_mesh, num_objects * sizeof(TriangleMesh)));

        // Allocate memory for the number of triangles in each mesh
        h_num_triangles = new int[num_objects];
        CUDA_CHECK(cudaMalloc(&d_num_triangles, num_objects * sizeof(int)));
    }
}

__host__ void SceneManager::freeResources() {
    // Free host memory
    if (h_image) {
        delete[] h_image;
        h_image = nullptr;
    }

    if (h_aabb) {
        delete[] h_aabb;
        h_aabb = nullptr;
    }

    // Delete the triangle mesh array if it exists
    if (h_mesh) {
        delete[] h_mesh;
        h_mesh = nullptr;
    }

    // Delete the number of triangles array if it exists
    if (h_num_triangles) {
        delete[] h_num_triangles;
        h_num_triangles = nullptr;
    }

    // Free device memory
    if (d_mesh) {
        cudaFree(d_mesh);
        d_mesh = nullptr;
    }

    if (d_image) {
        cudaFree(d_image);
        d_image = nullptr;
    }

    if (d_aabb) {
        cudaFree(d_aabb);
        d_aabb = nullptr;
    }

    if (d_num_triangles) {
        cudaFree(d_num_triangles);
        d_num_triangles = nullptr;
    }
}

// Copy data to device
__host__ void SceneManager::copyToDevice() {
    // Copy AABB data to device
    if (num_objects > 0 && h_aabb != nullptr) {
        // Copy the AABB's array data to device
        h_aabb->AABBMemcpyHtD();

        // Copy the AABB object to device
        CUDA_CHECK(
            cudaMemcpy(d_aabb, h_aabb, sizeof(AABB), cudaMemcpyHostToDevice));

        // Copy the entire mesh array to device
        if (d_mesh != nullptr && h_mesh != nullptr) {
            CUDA_CHECK(cudaMemcpy(d_mesh, h_mesh,
                                  num_objects * sizeof(TriangleMesh),
                                  cudaMemcpyHostToDevice));
        }

        // Copy the number of triangles to device
        if (d_num_triangles != nullptr && h_num_triangles != nullptr) {
            CUDA_CHECK(cudaMemcpy(d_num_triangles, h_num_triangles,
                                  num_objects * sizeof(int),
                                  cudaMemcpyHostToDevice));
        }
    }
}

// Copy data from device
__host__ void SceneManager::copyFromDevice() {
    // Copy rendered image from device
    CUDA_CHECK(cudaMemcpy(h_image, d_image, width * height * sizeof(Vec3),
                          cudaMemcpyDeviceToHost));
}

// Save image to file
__host__ void SceneManager::saveImage(const char* filename) {
    writeToPPM(filename, h_image, width, height);
    std::cout << "Image saved to " << filename << std::endl;
}

__host__ void SceneManager::renderBox() {
    // Get camera parameters from the stored reference
    CUDACameraParams camera_params = camera.CUDAparams();

    // Define grid and block dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);

    // Launch kernel
    Timer timer;
    timer.start("Rendering Scene");
    renderBoxKernel<<<grid_dim, block_dim>>>(d_image, camera_params, d_aabb,
                                             num_objects);
    cudaDeviceSynchronize();
    timer.stop();
}

__host__ void SceneManager::renderMesh() {
    // Get camera parameters from the stored reference
    CUDACameraParams camera_params = camera.CUDAparams();

    // Define grid and block dimensions
    dim3 block_dim(32, 4); // Adjusted for better warp scheduling / occupancy
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);

    Timer timer;
    timer.start("Rendering Scene");
    renderMeshKernel<<<grid_dim, block_dim>>>(d_image, d_aabb, d_mesh, 
                                              num_objects, d_num_triangles, camera_params);
    cudaDeviceSynchronize();
    timer.stop();
    CUDA_CHECK(cudaGetLastError());
}