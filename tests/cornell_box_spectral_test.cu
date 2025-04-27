#include <string>
#include <vector>

#include "camera.h"
#include "scene_manager.cuh"
#include "timer.h"
#include "util/spectrum.cuh"
#include "material.cuh"

// Spectral material IDs
enum ReflectanceID {
    WHITE = 0,
    RED = 1,
    GREEN = 2,
    LIGHT = 3
};

enum EmissionID {
    LIGHT_EMISSION
};

int main() {
    Timer timer;
    timer.start("Total Time");
    
    // Define list of objects here
    std::vector<std::string> obj_list = {
        "light",      "donkey_kong", "left_wall", "back_wall",
        "right_wall", "ceiling",     "floor"};
    int list_size = obj_list.size();

    // Define spectral material assignments
    std::vector<int> spectral_reflectance_ids = {
        LIGHT, WHITE, RED, WHITE, GREEN, WHITE, WHITE
    };
    
    // Define emission status
    std::vector<bool> is_emissive = {
        true, false, false, false, false, false, false
    };

    // Define emission spectrum IDs
    std::vector<int> spectral_emission_ids = {
        LIGHT_EMISSION, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE
    };

    // Camera setup with Cornell box parameters
    Camera camera(cornell_box_params);
    SceneManager scene(camera, list_size);
    
    // Load all objects
    for (int i = 0; i < list_size; i++) {
        scene.addTriangleMeshSpectrum(
            std::string(PROJECT_ROOT) + "/assets/cornell_box/" + obj_list[i] + ".obj",
            spectral_reflectance_ids[i],
            spectral_emission_ids[i],
            is_emissive[i],
            i
        );
        
        // Print material info
        std::cout << "Object " << i << " (" << obj_list[i] << "): "
                  << "Reflectance ID: " << spectral_reflectance_ids[i]
                  << ", Emission ID: " << spectral_emission_ids[i]
                  << ", Emissive: " << (is_emissive[i] ? "Yes" : "No") << std::endl;
    }

    // Render with multiple different sample counts
    std::vector<int> sample_counts = {1, 4, 16, 64, 256};
    
    for (int samples : sample_counts) {
        std::cout << "\nRendering with " << samples << " samples per pixel..." << std::endl;
        scene.renderSpectralMesh(samples);
        scene.copyFromDevice();
        // Save with sample count in filename
        std::string filename = "cornell_box_spectral_" + std::to_string(samples) + "spp.ppm";
        scene.saveImage(filename.c_str());
        std::cout << "Image saved as " << filename << std::endl;
    }
    
    // Single Render with x samples
    // int samples = 1024;
    // std::cout << "\nRendering with " << samples << " samples per pixel..." << std::endl;
    // scene.renderSpectralMesh(samples);
    // scene.copyFromDevice();
    // // Save with sample count in filename
    // std::string filename = "cornell_box_spectral_" + std::to_string(samples) + "spp.ppm";
    // scene.saveImage(filename.c_str());
    // std::cout << "Image saved as " << filename << std::endl;

    timer.stop();
    std::cout << "All renders completed successfully." << std::endl;
    return 0;
}
