#ifndef CRT_CUH
#define CRT_CUH

#include <cuda_runtime.h>

#include "material.cuh"
#include "ray_color.cuh"
#include "scene_manager.cuh"
#include "triangle_mesh.cuh"
#include "util/spectrum.cuh"

// Only declare as extern, not define here (it's defined in scene_manager.cu)
extern __constant__ Material c_materials[256];

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
    float dot_dir_sce1 =
        ray_dx * scrossex + ray_dy * scrossey + ray_dz * scrossez;
    v = inv_determinant * dot_dir_sce1;

    // Out of bounds
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    // Compute t to find intersection point - dot(edge2, s_cross_e1)
    float dot_e2_sce1 =
        edge2x * scrossex + edge2y * scrossey + edge2z * scrossez;
    t = inv_determinant * dot_e2_sce1;

    if (t > epsilon) {
        // Calculate intersection point directly
        intersection =
            Vec3(ray_ox + ray_dx * t, ray_oy + ray_dy * t, ray_oz + ray_dz * t);
        return true;
    }
    return false;
}

// Color ray based on hit triangle
// FIRST CHECK AABB, THEN CHECK TRIANGLE
__device__ Vec3 colorRayTriangle(const Ray& ray, const AABB* boxes,
                                 const TriangleMesh* mesh, int num_objects,
                                 const int* num_triangles) {
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

        return Vec3(c_materials[hit_obj_id].albedo.x,
                    c_materials[hit_obj_id].albedo.y,
                    c_materials[hit_obj_id].albedo.z);
        // return normalMap(normal);  // Color based on normal
        // Simple coloring based on object ID and triangle index
        // return threeColor(hit_obj_id);
    }

    // No hit, return sky background
    // return skyBg(ray);
    // No hit, return black
    return Vec3(0.0f, 0.0f, 0.0f);
}

__device__ Vec3 colorRayBox(const Ray& ray, const AABB* boxes,
                            int num_objects) {
    // Check intersection with any box in the array
    for (int i = 0; i < num_objects; i++) {
        if (boxes->hitAABB(ray, i)) {
            // posterized color based on obj id
            return quantizedColor(i);
        }
    }
    // Background color (gradient from blue to white)
    // return skyBg(ray);
    return Vec3(0.0f, 0.0f, 0.0f);  // No hit, return black
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
__global__ void renderMeshKernel(Vec3* image_buffer, AABB* boxes,
                                 TriangleMesh* meshes, const int num_objects,
                                 const int* __restrict__ num_triangles,
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
        image_buffer[pixel_idx] =
            colorRayTriangle(ray, boxes, meshes, num_objects, num_triangles);
    }
}

// Get material reflectance spectrum based on ID
__device__ SampledSpectrum getMaterialReflectanceSpectrum(int reflectance_id) {
    switch (reflectance_id) {
        case 0:  // WHITE
            return SampledSpectrum::WhiteReflectance();
        case 1:  // RED
            return SampledSpectrum::RedReflectance();
        case 2:  // GREEN
            return SampledSpectrum::GreenReflectance();
        case 3:  // LIGHT
            return SampledSpectrum::LightReflectance();
        default:
            return SampledSpectrum::WhiteReflectance();
    }
}

// Get material emission spectrum based on ID
__device__ SampledSpectrum getMaterialEmissionSpectrum(int emission_id) {
    switch (emission_id) {
        default:
            return SampledSpectrum::LightEmission();
    }
}

// TODO: Replace temp with light sampler stuff
// Random number generator (Xorshift) normalized to [0,1]
// https://en.wikipedia.org/wiki/Xorshift
__device__ float randomFloat(unsigned int* seed) {
    // Xorshift algorithm
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return static_cast<float>(*seed) / 4294967295.0f;  // Divide by 2^32-1
}

// Malley's Method: Generate cosine weighted point on hemisphere
// https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#Cosine-WeightedHemisphereSampling
__device__ Vec3 randomHemispherePoint(const Vec3& normal, unsigned int* seed) {
    // Generate random spherical coordinates
    float r = sqrtf(randomFloat(seed));
    float phi = 2.0f * M_PI * randomFloat(seed);
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    float z = sqrtf(fmaxf(0.0f, 1.0f - x * x - y * y));
    // Create orthonormal basis (u, v, w) from normal
    Vec3 w = normal;
    Vec3 u;

    // Find a vector perpendicular to w
    if (fabsf(w.x()) > 0.1f)
        u = cross(Vec3(0.0f, 1.0f, 0.0f), w);
    else
        u = cross(Vec3(1.0f, 0.0f, 0.0f), w);

    u = unit_vector(u);
    Vec3 v = cross(w, u);

    // Convert from local coordinates to world coordinates
    return u * x + v * y + w * z;
}

// Handle spectral ray intersection with triangles
__device__ bool spectralTracePath(const Ray& primary_ray, const AABB* boxes,
                                  const TriangleMesh* mesh, int num_objects,
                                  const int* num_triangles,
                                  const SampledWavelength& wavelength,
                                  SampledSpectrum& spectrum, unsigned int* seed,
                                  int bounce_limit = 5) {
    Ray current_ray = primary_ray;
    SampledSpectrum throughput = SampledSpectrum::SingleWavelength(wavelength);

    for (int bounce = 0; bounce < bounce_limit; bounce++) {
        float closest_t = INFINITY;
        int hit_obj_id = -1;
        int hit_tri_idx = -1;
        Vec3 hit_point;
        Vec3 hit_normal;
        float hit_u, hit_v;

        // Find closest intersection
        for (int obj_id = 0; obj_id < num_objects; obj_id++) {
            // First check AABB hit to avoid unnecessary triangle checks
            if (boxes->hitAABB(current_ray, obj_id)) {
                // Check all triangles in this mesh
                for (int tri_idx = 0; tri_idx < num_triangles[obj_id];
                     tri_idx++) {
                    Vec3 intersection;
                    float u, v, t;

                    if (rayIntersectsTriangle(current_ray, mesh, obj_id,
                                              tri_idx, intersection, u, v, t)) {
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

        // If we didn't hit anything, return the background
        if (hit_obj_id < 0) {
            return false;
        }

        // Calculate the surface normal at hit point
        float w = 1.0f - hit_u - hit_v;  // Third barycentric coordinate
        hit_normal =
            unit_vector(Vec3(w * mesh[hit_obj_id].d_n0x[hit_tri_idx] +
                                 hit_u * mesh[hit_obj_id].d_n1x[hit_tri_idx] +
                                 hit_v * mesh[hit_obj_id].d_n2x[hit_tri_idx],
                             w * mesh[hit_obj_id].d_n0y[hit_tri_idx] +
                                 hit_u * mesh[hit_obj_id].d_n1y[hit_tri_idx] +
                                 hit_v * mesh[hit_obj_id].d_n2y[hit_tri_idx],
                             w * mesh[hit_obj_id].d_n0z[hit_tri_idx] +
                                 hit_u * mesh[hit_obj_id].d_n1z[hit_tri_idx] +
                                 hit_v * mesh[hit_obj_id].d_n2z[hit_tri_idx]));

        // Ensure normal points against the ray
        if (dot(hit_normal, current_ray.direction()) > 0.0f) {
            hit_normal = hit_normal * -1.0f;
        }

        // Get the material properties of the hit object
        Material& mat = c_materials[hit_obj_id];

        // Handle emissive materials
        if (mat.is_emissive) {
            // Sample the emission spectrum
            SampledSpectrum emission =
                getMaterialEmissionSpectrum(mat.spectral_emission_id);
            // Multiply by the accumulated throughput
            spectrum = emission * throughput;
            return true;
        }

        // Get the reflectance spectrum of this material
        SampledSpectrum reflectance =
            getMaterialReflectanceSpectrum(mat.spectral_reflectance_id);

        // Update throughput with the reflectance at this wavelength
        throughput = throughput * reflectance;

        // Russian roulette termination to avoid spending time on paths that
        // contribute little
        // if (bounce > 2) {
        //     // Probability of continuing the path
        //     float p = fmaxf(0.1f, fmaxf(reflectance.sample(wavelength), 0.0f));
        //     if (randomFloat(seed) > p) {
        //         break;
        //     }
        //     // Compensate for the random termination
        //     throughput = throughput * (1.0f / p);
        // }

        // Generate a new ray direction based on material properties
        Vec3 new_direction = randomHemispherePoint(hit_normal, seed);

        // TODO: Handle different material types
        // switch (mat.type) {
        //     case MaterialType::DIFFUSE:
        //         new_direction = randomHemispherePoint(hit_normal, seed);
        //         break;
        //     default:
        //         new_direction = randomHemispherePoint(hit_normal, seed);
        //         break;
        // }

        // new origin after triangle hit
        Vec3 new_origin = hit_point + hit_normal;

        // Update the ray for the next bounce
        current_ray = Ray(new_origin, new_direction);
    }

    // Path terminated without hitting a light
    return false;
}

// sRGB gamma 2.2. correction - accurately converts linear RGB to sRGB
// https://en.wikipedia.org/wiki/SRGB#Definition
__device__ float srgbGammaCorrection(float linear) {
    if (linear <= 0.0031308f)
        return 12.92f * linear;
    else
        return 1.055f * __powf(linear, 1.0f / 2.4f) - 0.055f;
}

// Spectral rendering kernel
__global__ void renderSpectralMeshKernel(Vec3* image_buffer, AABB* boxes,
                                         TriangleMesh* meshes,
                                         const int num_objects,
                                         const int* __restrict__ num_triangles,
                                         const CUDACameraParams camera_params,
                                         const int samples_per_pixel,
                                         unsigned int rand_seed,
                                         volatile int* progress) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= camera_params.pixel_width || row >= camera_params.pixel_height) {
        return;
    }

    const int pixel_idx = row * camera_params.pixel_width + col;

    // Initialize random seed for this pixel
    unsigned int seed = rand_seed + pixel_idx;

    // Accumulate color for this pixel
    float3 pixel_color = make_float3(0.0f, 0.0f, 0.0f);

    // For each sample
    for (int s = 0; s < samples_per_pixel; s++) {
        // Jitter the pixel location for anti-aliasing
        // by offsetting [-0.5, 0.5]
        float offset_u = randomFloat(&seed) - 0.5f;
        float offset_v = randomFloat(&seed) - 0.5f;

        // Calculate ray for this sample
        const Vec3 pixel_center =
            camera_params.pixel00_loc +
            ((col + offset_u) * camera_params.pixel_delta_u) +
            ((row + offset_v) * camera_params.pixel_delta_v);
        const Vec3 ray_direction =
            unit_vector(pixel_center - camera_params.center);

        Ray ray(camera_params.center, ray_direction);

        // Random wavelength sampling (Hero wavelength technique)
        float u = randomFloat(&seed);
        SampledWavelength wavelength = SampledSpectrum::SampleWavelength(u);

        // Trace the path and get the spectral contribution
        SampledSpectrum spd;
        if (spectralTracePath(ray, boxes, meshes, num_objects, num_triangles,
                              wavelength, spd, &seed)) {
            // Convert spectrum to RGB and accumulate and apply tone mapping
            float3 rgb = spd.toRGB();
            pixel_color.x += rgb.x;
            pixel_color.y += rgb.y;
            pixel_color.z += rgb.z;
        }
    }

    // Average the samples
    pixel_color.x /= samples_per_pixel;
    pixel_color.y /= samples_per_pixel;
    pixel_color.z /= samples_per_pixel;

    // Reinhard tone mapping
    // C = C / (1 + C)
    // Exposure tone map from OpenGL
    // C = 1 - exp(-C * Exposure) 
    pixel_color.x = 1.0f - expf(-pixel_color.x * 3.5f);
    pixel_color.y = 1.0f - expf(-pixel_color.y * 3.5f);
    pixel_color.z = 1.0f - expf(-pixel_color.z * 3.5f);

    // Apply gamma correction
    pixel_color.x = srgbGammaCorrection(pixel_color.x);
    pixel_color.y = srgbGammaCorrection(pixel_color.y);
    pixel_color.z = srgbGammaCorrection(pixel_color.z);

    // Clamp min to 0.0f
    pixel_color.x = fmaxf(pixel_color.x, 0.0f);
    pixel_color.y = fmaxf(pixel_color.y, 0.0f);
    pixel_color.z = fmaxf(pixel_color.z, 0.0f);

    // Store the final color
    image_buffer[pixel_idx] = Vec3(pixel_color.x, pixel_color.y, pixel_color.z);

    // Progress bar update
    // Update progress per row of blocks to minimize atomics
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Store this block's linear ID
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        
        // Only update progress when larger row of blocks completes
        if (block_id % 16 == 0) {  // Adjust the value based on your grid dimensions
            atomicAdd((int*)progress, 1);
        }

        // Update progress bar to 100% when all blocks are done for nice output
        if (block_id == gridDim.x * gridDim.y - 1) {
            // Set to final value to ensure completion
            atomicExch((int*)progress, (gridDim.x * gridDim.y + 15) / 16);
        }

    }
}

#endif  // CRT_CUH