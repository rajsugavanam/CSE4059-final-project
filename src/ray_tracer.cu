#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "crt.cuh"
#include "ray.cuh"
#include "timer.h"
#include "vec3.cuh"
#include "ObjReader.cuh"

// perhaps move all this to a struct...
const float aspect_ratio = 16.0f / 9.0f;
const int pixel_height = 1080;
const int pixel_width = static_cast<int>(pixel_height * aspect_ratio);
const float focal_length = 1.0f;
const float viewport_height = 2.0f;
const float viewport_width =
    viewport_height * (float(pixel_width) / pixel_height);
const Vec3 camera_center = Vec3(0.0f, -6.0f, 7.0f); // the sphere is at a weird location

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

__device__ bool rayIntersectsTriangle(const Ray& ray, const Triangle3& triangle,
                                      Vec3& intersection, float& u, float& v); 

__device__ Vec3 rayColor(const Ray& ray, Triangle3* triangles,
                         int num_triangle);

__global__ void rayRender(Vec3* image_buffer, int width, int height,
                          Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
                          Vec3 camera_origin, Triangle3* triangle_mesh,
                          int num_triangles);

__global__ void greenRedRender(Vec3* image_buffer, int width, int height,
                               Vec3 pixel00_loc, Vec3 delta_u, Vec3 delta_v,
                               Vec3 camera_origin);

// Fucntion declaserioansen
void writeToPPM(const char* filename, Vec3* image_buffer, int width,
                int height);

int main() {
    // ===================
    // ===== THE OBJ =====
    // ===================
    Timer timer;
    timer.start("Loading OBJ file");
    
    // abs path to obj using PROJECT_ROOT macro in cmake 
    ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/sphere.obj");
    
    reader.readModel();
    Model model = reader.parsedModel;

    // Check for empty triangles instead of faces
    if (model.modelTriangles.empty()) {
        return 1;
    }
    timer.stop();


    // ========================
    // ===== MEMORY TRAIN =====
    // ========================
    timer.start("Memory Allocation on GPU");
    // init rgb image output buffer
    Vec3* image_buffer_h{new Vec3[image_buffer_size]()};
    Vec3* image_buffer_d;

    // load all the triangles
    Triangle3* triangles_h = model.modelTriangles.data();
    Triangle3* triangles_d;
    size_t size = model.modelTriangles.size();

    size_t trianglesMemSize = size*sizeof(Triangle3);
    cudaMalloc((void**)&image_buffer_d, image_buffer_byte_size);
    cudaMalloc(&triangles_d, trianglesMemSize);
    cudaMemcpy(image_buffer_d, image_buffer_h, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(triangles_d, triangles_h, trianglesMemSize, cudaMemcpyHostToDevice);
    timer.stop();

    // ===========================
    // ===== RUNNING ON GPU ======
    // ===========================
    timer.start("Kernel Launching");
    dim3 block_size(16, 16);
    dim3 grid_size((pixel_width + block_size.x - 1) / block_size.x,
                   (pixel_height + block_size.y - 1) / block_size.y);
    
    // Optimization 5: Add CUDA error checking
    rayRender<<<grid_size, block_size>>>(
        image_buffer_d, pixel_width, pixel_height, pixel00_loc, pixel_delta_u,
        pixel_delta_v, camera_center, triangles_d, size);
    
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
    cudaFree(image_buffer_d);
    cudaFree(triangles_d);

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