#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "crt.cuh"
// #include "ray.cuh"
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
const float camera_x = 0.0f;
const float camera_y = 0.0f;
const float camera_z = 0.0f;
// const Vec3 camera_center = Vec3(0.0f, -6.0f, 7.0f); // the sphere is at a weird location
const Vec3 camera_center = Vec3(camera_x, camera_y, camera_z);

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
    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/TyDonkeyKR.obj");
    ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/Untitled.obj");
    // ObjReader reader = ObjReader(std::string(PROJECT_ROOT) + "/assets/sphere.obj");
    
    reader.readModel();
    Model model = reader.parsedModel;

    // Check for empty triangles instead of faces
    if (model.modelTriangles.empty()) {
        return 1;
    }
    timer.stop();

    //  =================
    //  ===== WALLS =====
    //  =================
    size_t num_walls{1};
    float z_length = -10.0f;
    float y_length = -1.0f;
    float x_length = -z_length * 0.5f;
    int num_wall_tri = 2;
    Triangle3* walls_h{new Triangle3[num_wall_tri]{
            Triangle3{
                Vec3{-x_length, y_length, z_length},
                Vec3{x_length, y_length, z_length},
                Vec3{x_length, y_length, z_length * 1.5f}
            },
            Triangle3{
                Vec3{-x_length, y_length, z_length},
                Vec3{-x_length, y_length, z_length * 1.5f},
                Vec3{x_length, y_length, z_length * 1.5f}
            }
        }
    };


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
    size_t mesh_size = model.modelTriangles.size();
    size_t trianglesMemSize = mesh_size * sizeof(Triangle3);
    Triangle3* walls_d;
    size_t wall_size = 2 * num_walls * sizeof(Triangle3);
    std::cout << "wall byte size: " << wall_size << "\n"
             << "Vec3 byte size: " << sizeof(Vec3) << "\n"
             << "Triangle3 byte size: " << sizeof(Triangle3) << std::endl;

    // from cuRAND
    curandState *rngStates_d = 0;

    // malloc state
    cudaMalloc((void**)&image_buffer_d, image_buffer_byte_size);
    cudaMalloc((void**)&triangles_d, trianglesMemSize);
    cudaMalloc((void**)&rngStates_d, mesh_size*sizeof(*rngStates_d));
    cudaMalloc((void**)&walls_d, wall_size);
    cudaMemcpy(image_buffer_d, image_buffer_h, image_buffer_byte_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(triangles_d, triangles_h, trianglesMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(walls_d, walls_h, wall_size, cudaMemcpyHostToDevice);
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
        pixel_delta_v, camera_center, triangles_d, mesh_size, walls_d, num_wall_tri);
    
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
    delete[] walls_h;
    cudaFree(image_buffer_d);
    cudaFree(triangles_d);
    cudaFree(walls_d);
    cudaFree(rngStates_d);

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
