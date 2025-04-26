#ifndef CRT_CUH
#define CRT_CUH

#include <cuda_runtime.h>
#include "scene_manager.cuh"
#include "material.cuh"
#include "triangle_mesh.cuh"
#include "ray_color.cuh"

__constant__ Material c_materials[256];

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

        return Vec3(c_materials[hit_obj_id].albedo.x, c_materials[hit_obj_id].albedo.y, c_materials[hit_obj_id].albedo.z);
        // return normalMap(normal);  // Color based on normal
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

#endif // CRT_CUH
