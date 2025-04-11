#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <math/VecN.cuh>
// #include <math/vec3.cuh>
#include <ostream>

#include "ray.cuh"
#include "timer.h"
#include "triangle3.cuh"
#include "vec3.cuh"

// perhaps move all this to a struct...
const float aspect_ratio = 16.0f / 9.0f;
const int pixel_height = 1080;
const int pixel_width = static_cast<int>(pixel_height * aspect_ratio);
const float focal_length = 1.0f;
const float viewport_height = 2.0f;
const float viewport_width =
    viewport_height * (float(pixel_width) / pixel_height);
const Vec3 camera_center = Vec3(0.0f, 0.0f, 0.0f);

// viewport vector
const Vec3 viewport_u = Vec3(viewport_width, 0.0f, 0.0f);
const Vec3 viewport_v = Vec3(0.0f, -viewport_height, 0.0f);

const Vec3 pixel_delta_u = viewport_u / pixel_width;
const Vec3 pixel_delta_v = viewport_v / pixel_height;

const Vec3 viewport_upper_left = camera_center -
                                 Vec3(0.0f, 0.0f, focal_length) -
                                 viewport_u / 2 - viewport_v / 2;
const Vec3 pixel00_loc =
    viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
const int image_buffer_size = pixel_width * pixel_height;
const size_t image_buffer_byte_size = image_buffer_size * sizeof(Vec3);

// Möller–Trumbore intersection algorithm
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ bool rayIntersectsTriangle(const Ray& ray, const Triangle3& triangle,
                                      Vec3& intersection) {
    const float epsilon = 1.19209e-07f;

    Vec3 edge1 = triangle.edge0();
    Vec3 edge2 = triangle.edge1();
    Vec3 ray_cross_e2 = cross(ray.direction(), edge2);
    float determinant = dot(edge1, ray_cross_e2);

    // for parallel to triangle
    if (determinant > -epsilon && determinant < epsilon) {
        return false;
    }

    float inv_determinant = 1.0f / determinant;
    Vec3 s = ray.origin() - triangle.vertex0();
    float u = inv_determinant * dot(s, ray_cross_e2);

    // no nuggies
    if ((u < 0 && abs(u) > epsilon) || (u > 1 && abs(u - 1) > epsilon)) {
        return false;
    }

    Vec3 s_cross_e1 = cross(s, edge1);
    float v = inv_determinant * dot(ray.direction(), s_cross_e1);

    if ((v < 0 && abs(v) > epsilon) ||
        (u + v > 1 && abs(u + v - 1) > epsilon)) {
        return false;
    }

    // compute t to find interesection point
    float t = inv_determinant * dot(edge2, s_cross_e1);

    if (t > epsilon) {
        intersection = ray.origin() + ray.direction() * t;
        return true;
    } else {
        return false;
    }
}

// alpha blending with precise method
// normal map works for now... changing one z coord of triangle will correctly
// map the color!
__device__ Vec3 rayColor(const Ray& ray, Triangle3* triangles,
                         int num_triangle) {
    // Triangle3 triangle(Vec3(0.0f, 0.5f, -1.0f),     // top
    //                    Vec3(0.5f, -0.5f, -1.0f),    // right
    //                    Vec3(-0.5f, -0.5f, -1.0f));  // left
    // Vec3 intersection;
    Vec3 result{};
    for (int tri = 0; tri < num_triangle; tri++) {
        Triangle3 triangle = triangles[tri];
        Vec3 intersection;
        if (rayIntersectsTriangle(
                ray, triangle,
                intersection)) {  // return Vec3(1.0f, 0.0f, 0.0f); //
                                  // red for now... later we use
            // normals
            Vec3 normal =
                cross(triangle.edge0(),
                      triangle.edge1());  // we can probably place this moller
                                          // code, depends on obj stuff
            normal =
                unit_vector(-normal);  // the normal is pointing backwards rn...
            result = 0.5f * Vec3(normal.x() + 1.0f, normal.y() + 1.0f,
                                 normal.z() + 1.0f);
            break;
        }
        Vec3 unit_direction = unit_vector(ray.direction());
        float alpha =
            0.5f * (unit_direction.y() + 1.0);  // y = [-1,1] to y = [0,1]
        // lerp between white (1, 1, 1) to sky_blue (0.5, 0.7, 1)
        result = (1.0f - alpha) * Vec3(1.0f, 1.0f, 1.0f) +
                 alpha * Vec3(0.5f, 0.7f, 1.0f);
    }
    return result;
}

__global__ void rayRender(Vec3* image_buffer, int width, int height,
                          Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
                          Vec3 camera_origin, Triangle3* triangle_mesh,
                          int num_triangles) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int pixel_idx = row * width + col;  // 3 channels (RGB)

        Vec3 pixel_center = pixel00_loc + (col * delta_u) + (row * delta_v);
        Vec3 ray_direction = pixel_center - camera_origin;

        Ray ray(camera_origin, ray_direction);
        Vec3 pixel_color = rayColor(ray, triangle_mesh, num_triangles);
        image_buffer[pixel_idx] = pixel_color;
    }
}

__global__ void greenRedRender(Vec3* image_buffer, int width, int height,
                               Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
                               Vec3 camera_origin) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // int pixel_idx = (row * width + col) * 3;  // 3 channels (RGB)
        // image_buffer[pixel_idx] = float(col) / float(width);       // Red
        // image_buffer[pixel_idx + 1] = float(row) / float(height);  // Green
        // image_buffer[pixel_idx + 2] = 0.5f;                         // Blue

        // each thread computes all 3 channels
        int pixel_idx = row * width + col;
        image_buffer[pixel_idx] =
            Vec3(float(col) / float(width), float(row) / float(height), 0.5f);
    }
}

// Fucntion declaserioansen
void writeToPPM(const char* filename, Vec3* image_buffer, int width,
                int height);

void shittyMain() {
    Timer timer;
    // ========================
    // ===== MEMORY TRAIN =====
    // ========================
    timer.start("Memory Allocation on GPU");
    // init rgb image output buffer
    Vec3* image_buffer_h{new Vec3[image_buffer_size]()};
    Vec3* image_buffer_d;

    // init triangle mesh buffer
    unsigned int num_triangles{3};
    Triangle3* triangle_mesh_h{
        new Triangle3[num_triangles]{{
                                         // c tr
                                         Vec3(0.0f, 0.5f, -1.0f),   // top
                                         Vec3(0.5f, -0.5f, -1.0f),  // right
                                         Vec3(-0.5f, -0.5f, -1.0f)  // l
                                     },
                                     {
                                         // r tri
                                         Vec3(0.0f, 0.5f, -1.0f),   // t
                                         Vec3(1.5f, -0.5f, -2.0f),  // r
                                         Vec3(0.5f, -0.5f, -1.0f)   // l
                                     },                             // l
                                     {
                                         // l tri
                                         Vec3(0.0f, 0.5f, -1.0f),    // t
                                         Vec3(-0.5f, -0.5f, -1.0f),  // r
                                         Vec3(-1.5f, -0.5f, -2.0f)   // l
                                     }}};
    Triangle3* triangle_mesh_d;
    // malloc and cpy
    cudaMalloc((void**)&image_buffer_d, image_buffer_byte_size);
    cudaMalloc((void**)&triangle_mesh_d, sizeof(Triangle3) * num_triangles);
    cudaMemcpy(image_buffer_d, image_buffer_h, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(triangle_mesh_d, triangle_mesh_h,
               sizeof(Triangle3) * num_triangles, cudaMemcpyHostToDevice);
    timer.stop();

    // ===========================
    // ===== RUNNING ON GPU ======
    // ===========================
    timer.start("Kernel Launching");
    dim3 block_size(16, 16);
    dim3 grid_size((pixel_width + block_size.x - 1) / block_size.x,
                   (pixel_height + block_size.y - 1) / block_size.y);
    rayRender<<<grid_size, block_size>>>(
        image_buffer_d, pixel_width, pixel_height, pixel00_loc, pixel_delta_u,
        pixel_delta_v, camera_center, triangle_mesh_d, num_triangles);
    cudaDeviceSynchronize();
    timer.stop();

    // ============================
    // ===== CPU IMAGE WRITER =====
    // ============================
    timer.start("Copying back to host");
    cudaMemcpy(image_buffer_h, image_buffer_d, image_buffer_byte_size,
               cudaMemcpyDeviceToHost);
    timer.stop();

    timer.start("Outputting to PPM file");
    writeToPPM("outout.ppm", image_buffer_h, pixel_width, pixel_height);
    timer.stop();

    // ==========================
    // ===== MEMORY FREEDOM =====
    // ==========================
    delete[] image_buffer_h;
    delete[] triangle_mesh_h;
    cudaFree(image_buffer_d);
    cudaFree(triangle_mesh_d);
}

void notShittyMain() {

}

int main() {
    // shittyMain();
    notShittyMain();
    return 0;
}

void writeToPPM(const char* filename, Vec3* image_buffer, int pixel_width,
                int pixel_height) {
    std::ofstream os(filename);
    os << "P3\n" << pixel_width << " " << pixel_height << "\n255\n";
    for (int j = 0; j < pixel_height; j++) {
        for (int i = 0; i < pixel_width; i++) {
            int pixel_idx = (j * pixel_width + i);
            int r = static_cast<int>(image_buffer[pixel_idx].x() * 255.999);
            int g = static_cast<int>(image_buffer[pixel_idx].y() * 255.999);
            int b = static_cast<int>(image_buffer[pixel_idx].z() * 255.999);
            os << r << " " << g << " " << b << "\n";
        }
    }
    os.close();
}
