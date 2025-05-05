#include <string>
#include <vector>

#include "camera.h"
#include "material.cuh"
#include "scene_manager.cuh"
#include "timer.h"
#include "util/spectrum.cuh"


// Spectral material IDs
enum ReflectanceID { WHITE = 0, RED = 1, GREEN = 2, LIGHT = 3 };

enum EmissionID { LIGHT_EMISSION };

// TODO: clean up main => move to scene manager
// create new src/crt.cu with arg for number of samples per pixel
int main() {
    // Timer total_timer;
    // total_timer.start("Cornell Box Spectral Test");
    // Define list of objects here
    std::vector<std::string> obj_list = {"light",     "small_box", "large_box",
                                         "left_wall", "back_wall", "right_wall",
                                         "ceiling",   "floor"};
    int list_size = obj_list.size();

    // Define spectral material assignments
    std::vector<int> spectral_reflectance_ids = {LIGHT, WHITE, WHITE, RED,  WHITE,
                                                 GREEN, WHITE, WHITE};

    // Define emission status
    std::vector<bool> is_emissive = {true,  false, false, false, false,
                                     false, false, false};

    // Define emission spectrum IDs
    std::vector<int> spectral_emission_ids = {
        LIGHT_EMISSION, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE};

    // Camera setup with Cornell box parameters
    CameraParams cb_custom = cornell_box_params;
    cb_custom.pixel_height = 2880;
    Camera camera(cb_custom);
    SceneManager scene(camera, list_size);

    // Load all objects
    for (int i = 0; i < list_size; i++) {
        scene.addTriangleMeshSpectrum(
            std::string(PROJECT_ROOT) + "/assets/accurate_cb/" + obj_list[i] +
                ".obj",
            spectral_reflectance_ids[i], spectral_emission_ids[i],
            is_emissive[i], i);

        // Print material info
        std::cout << "Object " << i << " (" << obj_list[i] << "): "
                  << "Reflectance ID: " << spectral_reflectance_ids[i]
                  << ", Emission ID: " << spectral_emission_ids[i]
                  << ", Emissive: " << (is_emissive[i] ? "Yes" : "No")
                  << std::endl;
    }

    // Render with multiple different sample counts
    // std::vector<int> sample_counts = {1, 4, 16, 64, 256, 1024, 4096, 16384};

    // for (int samples : sample_counts) {
    //     std::cout << "\nRendering with " << samples << " samples per pixel..."
    //               << std::endl;
    //     scene.renderSpectralMesh(samples);
    //     scene.copyFromDevice();
    //     // Save with sample count in filename
    //     std::string filename =
    //         "cornell_box_spectral_" + std::to_string(samples) + "spp.ppm";
    //     scene.saveImage(filename.c_str());
    // }

    // Single Render with x samples
    int samples = 4;
    std::cout << "\nRendering with " << samples << " samples per pixel..." <<
    std::endl; scene.renderSpectralMesh(samples); scene.copyFromDevice();
    // Save with sample count in filename
    std::string filename = "cornell_box_spectral_" + std::to_string(samples)
    + "spp.ppm"; scene.saveImage(filename.c_str());

    // total_timer.stop();
    std::cout << "All renders completed successfully." << std::endl;
    return 0;
}
