#ifndef CAMERA_H
#define CAMERA_H

#include <fstream>

#include "vec3.cuh"

struct CUDACameraParams {
    int pixel_width;
    int pixel_height;
    Vec3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 center;
};

// TODO: Add CameraParam Struct for constructor
struct CameraParams {
    float aspect_ratio;
    int pixel_height;
    float vfov;
    Vec3 look_from;
    Vec3 look_at;
    Vec3 v_up;
};
// struct CameraParams {
//     float aspect_ratio{16.0f / 9.0f};
//     int pixel_height{1080};
//     // int samples_per_pixel{10};
//     // int max_depth{10};

//     // Camera positioning
//     float vfov{90.0f};  // vertical field of view [degrees]
//     Vec3 look_from{0.0f, 0.0f, 0.0f};
//     Vec3 look_at{0.0f, 0.0f, -1.0f};
//     Vec3 v_up{0.0f, 1.0f, 0.0f};
//     float focal_length{1.0f};
// };

// Cornell box CameraParams
// C++20 designated initializers...
// CameraParams cornell_box_params = {
//     .aspect_ratio{1.0f},
//     .pixel_height{1080},
//     .vfov{40.0f},
//     .look_from{278.0f, 278.0f, -800.0f},
//     .look_at{278.0f, 278.0f, 0.0f},
//     .v_up{0.0f, 1.0f, 0.0f},
// };

const CameraParams cornell_box_params = {
    1.0f,
    2160,
    40.0f,
    Vec3(278.0f, 278.0f, 800.0f),
    Vec3(278.0f, 278.0f, 0.0f),
    Vec3(0.0f, 1.0f, 0.0f),
};

class Camera {
   public:
    float aspect_ratio{16.0f / 9.0f};
    int pixel_height{1080};
    // int samples_per_pixel{10};
    // int max_depth{10};

    // Camera positioning
    float vfov{90.0f};  // vertical field of view [degrees]
    Vec3 look_from{0.0f, 0.0f, 0.0f};
    Vec3 look_at{0.0f, 0.0f, -1.0f};
    Vec3 v_up{0.0f, 1.0f, 0.0f};
    float focal_length{1.0f};

    Camera() = default;

    Camera(CameraParams params)
        : aspect_ratio(params.aspect_ratio),
          pixel_height(params.pixel_height),
          vfov(params.vfov),
          look_from(params.look_from),
          look_at(params.look_at),
          v_up(params.v_up) {
        initialize();
    }

    Camera(double aspect_ratio, int pixel_height, double vfov, Vec3 look_from,
           Vec3 look_at, Vec3 v_up)
        : aspect_ratio(aspect_ratio),
          pixel_height(pixel_height),
          vfov(vfov),
          look_from(look_from),
          look_at(look_at),
          v_up(v_up) {
        initialize();
    }

    // move cuda kernel params
    CUDACameraParams CUDAparams() {
        CUDACameraParams camera_params;

        camera_params.pixel_width = pixel_width;
        camera_params.pixel_height = pixel_height;
        camera_params.pixel00_loc = pixel00_loc;
        camera_params.pixel_delta_u = pixel_delta_u;
        camera_params.pixel_delta_v = pixel_delta_v;
        camera_params.center = center;

        return camera_params;
    }

   private:
    int pixel_width;
    Vec3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 center;
    Vec3 u, v, w;  // camera basis vectors

    void initialize() {
        pixel_width = static_cast<int>(pixel_height * aspect_ratio);
        center = look_from;

        // Camera basis vectors
        w = unit_vector(look_from - look_at);
        u = unit_vector(cross(v_up, w));
        v = cross(w, u);

        // Viewport dimensions
        // https://en.wikipedia.org/wiki/Field_of_view#Photography
        focal_length = (look_from - look_at).length();
        float theta = vfov * M_PI / 180.0;  // convert to radians
        float viewport_height = 2.0f * tan(theta * 0.5f) * focal_length;
        float viewport_width =
            viewport_height * (static_cast<float>(pixel_width) /
                               static_cast<float>(pixel_height));

        // Viewport vectors
        Vec3 viewport_u = viewport_width * u;
        Vec3 viewport_v = viewport_height * -v;  // points down from top left

        // Pixel delta vectors
        pixel_delta_u = viewport_u / static_cast<float>(pixel_width);
        pixel_delta_v = viewport_v / static_cast<float>(pixel_height);

        // Upper left corner of the viewport
        Vec3 viewport_upper_left =
            center - (focal_length * w) - 0.5f * viewport_u - 0.5f * viewport_v;
        pixel00_loc =
            viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
    }
};

// TODO: delete after implementing class
// outputs image to current working directory
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
#endif  // CAMERA_H