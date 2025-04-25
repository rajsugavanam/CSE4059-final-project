#include "obj_reader.cuh"
#include "reduction.cuh"
#include "triangle_mesh.cuh"

// TriangleMesh class definition

// Default constructor
__host__ TriangleMesh::TriangleMesh()
    : num_triangles(0),
      h_v0x(nullptr),
      h_v0y(nullptr),
      h_v0z(nullptr),
      h_v1x(nullptr),
      h_v1y(nullptr),
      h_v1z(nullptr),
      h_v2x(nullptr),
      h_v2y(nullptr),
      h_v2z(nullptr),
      h_n0x(nullptr),
      h_n0y(nullptr),
      h_n0z(nullptr),
      h_n1x(nullptr),
      h_n1y(nullptr),
      h_n1z(nullptr),
      h_n2x(nullptr),
      h_n2y(nullptr),
      h_n2z(nullptr),
      d_v0x(nullptr),
      d_v0y(nullptr),
      d_v0z(nullptr),
      d_v1x(nullptr),
      d_v1y(nullptr),
      d_v1z(nullptr),
      d_v2x(nullptr),
      d_v2y(nullptr),
      d_v2z(nullptr),
      d_n0x(nullptr),
      d_n0y(nullptr),
      d_n0z(nullptr),
      d_n1x(nullptr),
      d_n1y(nullptr),
      d_n1z(nullptr),
      d_n2x(nullptr),
      d_n2y(nullptr),
      d_n2z(nullptr) {}

// Constructor with number of triangles
__host__ TriangleMesh::TriangleMesh(int num_triangles)
    : num_triangles(num_triangles) {
    mallocTriangleMesh();
    cudaMallocTriangleMesh();
}

__host__ TriangleMesh::~TriangleMesh() {
    freeTriangleMesh();
    cudaFreeTriangleMesh();

    std::cout << "TriangleMesh destructor called" << std::endl;
}

// Allocate host memory
__host__ void TriangleMesh::mallocTriangleMesh() {
    if (num_triangles > 0) {
        // Vertices
        h_v0x = new float[num_triangles];
        h_v0y = new float[num_triangles];
        h_v0z = new float[num_triangles];
        h_v1x = new float[num_triangles];
        h_v1y = new float[num_triangles];
        h_v1z = new float[num_triangles];
        h_v2x = new float[num_triangles];
        h_v2y = new float[num_triangles];
        h_v2z = new float[num_triangles];

        // Normals
        h_n0x = new float[num_triangles];
        h_n0y = new float[num_triangles];
        h_n0z = new float[num_triangles];
        h_n1x = new float[num_triangles];
        h_n1y = new float[num_triangles];
        h_n1z = new float[num_triangles];
        h_n2x = new float[num_triangles];
        h_n2y = new float[num_triangles];
        h_n2z = new float[num_triangles];
    }
}

// Free host memory
__host__ void TriangleMesh::freeTriangleMesh() {
    // Vertices
    delete[] h_v0x;
    delete[] h_v0y;
    delete[] h_v0z;
    delete[] h_v1x;
    delete[] h_v1y;
    delete[] h_v1z;
    delete[] h_v2x;
    delete[] h_v2y;
    delete[] h_v2z;

    // Normals
    delete[] h_n0x;
    delete[] h_n0y;
    delete[] h_n0z;
    delete[] h_n1x;
    delete[] h_n1y;
    delete[] h_n1z;
    delete[] h_n2x;
    delete[] h_n2y;
    delete[] h_n2z;

    h_v0x = h_v0y = h_v0z = nullptr;
    h_v1x = h_v1y = h_v1z = nullptr;
    h_v2x = h_v2y = h_v2z = nullptr;
    h_n0x = h_n0y = h_n0z = nullptr;
    h_n1x = h_n1y = h_n1z = nullptr;
    h_n2x = h_n2y = h_n2z = nullptr;
}

// Allocate device memory
__host__ void TriangleMesh::cudaMallocTriangleMesh() {
    if (num_triangles > 0) {
        // Vertices
        cudaMalloc(&d_v0x, num_triangles * sizeof(float));
        cudaMalloc(&d_v0y, num_triangles * sizeof(float));
        cudaMalloc(&d_v0z, num_triangles * sizeof(float));
        cudaMalloc(&d_v1x, num_triangles * sizeof(float));
        cudaMalloc(&d_v1y, num_triangles * sizeof(float));
        cudaMalloc(&d_v1z, num_triangles * sizeof(float));
        cudaMalloc(&d_v2x, num_triangles * sizeof(float));
        cudaMalloc(&d_v2y, num_triangles * sizeof(float));
        cudaMalloc(&d_v2z, num_triangles * sizeof(float));

        // Normals
        cudaMalloc(&d_n0x, num_triangles * sizeof(float));
        cudaMalloc(&d_n0y, num_triangles * sizeof(float));
        cudaMalloc(&d_n0z, num_triangles * sizeof(float));
        cudaMalloc(&d_n1x, num_triangles * sizeof(float));
        cudaMalloc(&d_n1y, num_triangles * sizeof(float));
        cudaMalloc(&d_n1z, num_triangles * sizeof(float));
        cudaMalloc(&d_n2x, num_triangles * sizeof(float));
        cudaMalloc(&d_n2y, num_triangles * sizeof(float));
        cudaMalloc(&d_n2z, num_triangles * sizeof(float));
    }
}

// CUDA device memory management
__host__ void TriangleMesh::meshMemcpyHtD() {
    // Safety check - don't try to copy if no triangles or memory not allocated
    if (num_triangles <= 0 || h_v0x == nullptr || d_v0x == nullptr) {
        std::cerr << "Warning: Cannot copy mesh data - no triangles or memory "
                     "not allocated"
                  << std::endl;
        return;
    }

    // Copy data to device
    cudaMemcpy(d_v0x, h_v0x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0y, h_v0y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v0z, h_v0z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1x, h_v1x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1y, h_v1y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1z, h_v1z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2x, h_v2x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2y, h_v2y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2z, h_v2z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);

    // Copy normals
    cudaMemcpy(d_n0x, h_n0x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n0y, h_n0y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n0z, h_n0z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n1x, h_n1x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n1y, h_n1y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n1z, h_n1z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2x, h_n2x, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2y, h_n2y, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_n2z, h_n2z, num_triangles * sizeof(float),
               cudaMemcpyHostToDevice);
}

__host__ void TriangleMesh::cudaFreeTriangleMesh() {
    cudaFree(d_v0x);
    cudaFree(d_v0y);
    cudaFree(d_v0z);
    cudaFree(d_v1x);
    cudaFree(d_v1y);
    cudaFree(d_v1z);
    cudaFree(d_v2x);
    cudaFree(d_v2y);
    cudaFree(d_v2z);
    cudaFree(d_n0x);
    cudaFree(d_n0y);
    cudaFree(d_n0z);
    cudaFree(d_n1x);
    cudaFree(d_n1y);
    cudaFree(d_n1z);
    cudaFree(d_n2x);
    cudaFree(d_n2y);
    cudaFree(d_n2z);
}

// Load a mesh from an OBJ file
__host__ TriangleMesh* TriangleMesh::loadFromOBJ(const std::string& filename) {
    std::cout << "Loading mesh from OBJ file: " << filename << std::endl;

    // Use the existing ObjReader to parse the file
    ObjReader reader(filename);
    reader.readModel();

    // Create a new mesh from the parsed triangles
    TriangleMesh* mesh = new TriangleMesh();

    // Get the triangles from the model
    const std::vector<Triangle3>& triangles = reader.parsedModel.modelTriangles;
    int numTriangles = triangles.size();

    if (numTriangles == 0) {
        std::cerr << "Error: No triangles found in OBJ file" << std::endl;
        delete mesh;
        return nullptr;
    }

    std::cout << "Loaded " << numTriangles << " triangles" << std::endl;

    // Initialize the mesh with the correct triangle count
    mesh->num_triangles = numTriangles;
    mesh->mallocTriangleMesh();

    // Copy the triangle data
    for (int i = 0; i < numTriangles; i++) {
        // Copy vertices
        mesh->h_v0x[i] = triangles[i].vertex0().x();
        mesh->h_v0y[i] = triangles[i].vertex0().y();
        mesh->h_v0z[i] = triangles[i].vertex0().z();

        mesh->h_v1x[i] = triangles[i].vertex1().x();
        mesh->h_v1y[i] = triangles[i].vertex1().y();
        mesh->h_v1z[i] = triangles[i].vertex1().z();

        mesh->h_v2x[i] = triangles[i].vertex2().x();
        mesh->h_v2y[i] = triangles[i].vertex2().y();
        mesh->h_v2z[i] = triangles[i].vertex2().z();

        // Copy normals
        mesh->h_n0x[i] = triangles[i].normal0().x();
        mesh->h_n0y[i] = triangles[i].normal0().y();
        mesh->h_n0z[i] = triangles[i].normal0().z();

        mesh->h_n1x[i] = triangles[i].normal1().x();
        mesh->h_n1y[i] = triangles[i].normal1().y();
        mesh->h_n1z[i] = triangles[i].normal1().z();

        mesh->h_n2x[i] = triangles[i].normal2().x();
        mesh->h_n2y[i] = triangles[i].normal2().y();
        mesh->h_n2z[i] = triangles[i].normal2().z();
    }

    // Allocate device memory
    mesh->cudaMallocTriangleMesh();

    return mesh;
}

// Enhanced version of computeAABB that uses stream-based reduction
__host__ void TriangleMesh::computeAABB(AABB* aabb, int obj_id) {
    // If the mesh is empty or aabb is null, return early
    if (num_triangles <= 0 || aabb == nullptr) {
        std::cerr << "Warning: Empty mesh or null AABB in computeAABB"
                  << std::endl;
        return;
    }

    // Allocate arrays for min and max bounds
    float min_bounds[3] = {INFINITY, INFINITY, INFINITY};
    float max_bounds[3] = {-INFINITY, -INFINITY, -INFINITY};

    // Use the stream-based reduction function to compute bounds
    computeMeshReductionStreams(this, obj_id, min_bounds, max_bounds);

    // Set the min/max values for the bounding box at the specified object ID
    aabb->h_minx[obj_id] = min_bounds[0];
    aabb->h_miny[obj_id] = min_bounds[1];
    aabb->h_minz[obj_id] = min_bounds[2];
    aabb->h_maxx[obj_id] = max_bounds[0];
    aabb->h_maxy[obj_id] = max_bounds[1];
    aabb->h_maxz[obj_id] = max_bounds[2];

    std::cout << "AABB computed for object " << obj_id << ": (" << min_bounds[0]
              << ", " << min_bounds[1] << ", " << min_bounds[2] << ") to ("
              << max_bounds[0] << ", " << max_bounds[1] << ", " << max_bounds[2]
              << ")" << std::endl;
}

// Get pointers to raw vertex data for direct use in kernels
__host__ void TriangleMesh::getRawVertexPointers(
    const float** v0x, const float** v0y, const float** v0z, const float** v1x,
    const float** v1y, const float** v1z, const float** v2x, const float** v2y,
    const float** v2z) const {
    *v0x = h_v0x;
    *v0y = h_v0y;
    *v0z = h_v0z;
    *v1x = h_v1x;
    *v1y = h_v1y;
    *v1z = h_v1z;
    *v2x = h_v2x;
    *v2y = h_v2y;
    *v2z = h_v2z;
}

// Get pointers to device vertex data for direct use in kernels
__host__ void TriangleMesh::getDeviceVertexPointers(
    float** v0x, float** v0y, float** v0z,
    float** v1x, float** v1y, float** v1z,
    float** v2x, float** v2y, float** v2z) const {
    
    *v0x = d_v0x;
    *v0y = d_v0y;
    *v0z = d_v0z;
    *v1x = d_v1x;
    *v1y = d_v1y;
    *v1z = d_v1z;
    *v2x = d_v2x;
    *v2y = d_v2y;
    *v2z = d_v2z;
}