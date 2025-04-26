// #include <fstream>
#include <iostream>

#include "camera.h"
#include "cuda_helper.h"
#include "image_writer.h"
#include "model.cuh"
#include "obj_reader.cuh"
// #include "ray_color.cuh"
#include "scene_manager.cuh"
#include "timer.h"
#include "triangle_mesh.cuh"
#include "crt.cuh"

extern __constant__ Material c_materials[256];

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

__host__ void SceneManager::addTriangleMeshColor(const std::string& filename, float3 albedo,
                                            int obj_id) {
    addTriangleMesh(filename, obj_id);
    materials[obj_id].albedo = albedo;
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

    CUDA_CHECK(cudaMemcpyToSymbol(c_materials, materials,
        sizeof(Material) * num_objects));

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
