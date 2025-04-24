#include <iostream>

#include "camera.h"
#include "cuda_helper.h"
#include "ray.cuh"
#include "scene_manager.cuh"
#include "timer.h"
#include "vec3.cuh"

// AABB render with scene manager
int test_scene_manager() {
    try {
        // init timer
        Timer timer;

        // Init Camera first
        CameraParams params;
        Camera camera(params);

        // init scene manager with the specified dimensions and camera
        SceneManager scene(camera, 2);

        // Add an AABB box with the same dimensions as in the aabb_raii_test
        scene.addAABB(-0.5f, -0.5f, -2.0f,    // min point (x,y,z)
                      0.5f, 0.5f, -1.0f, 0);  // max point (x,y,z)

        // add second AABB box
        scene.addAABB(-1.0f, -1.0f, -2.0f,    // min point (x,y,z)
                      1.0f, 1.0f, -1.0f, 1);  // max point (x,y,z)

        std::cout << "Copying data to device..." << std::endl;
        
        // Make sure AABB data is set up properly
        scene.copyToDevice();

        // Report we're starting rendering
        std::cout << "Starting render..." << std::endl;
        
        // Render the scene
        scene.render();

        std::cout << "Copying from device..." << std::endl;
        
        // Copy data back from device
        scene.copyFromDevice();

        // Save the image
        scene.saveImage("scene_manager_test.ppm");

        std::cout << "Scene Manager test completed. Output saved to "
                     "scene_manager_test.ppm"
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

// Main function to be called from test runner
int main(int argc, char** argv) { return test_scene_manager(); }