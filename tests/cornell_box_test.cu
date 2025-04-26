#include <string>
#include <vector>

#include "camera.h"
#include "scene_manager.cuh"
#include "timer.h"
// #include "cuda_helper.h"

extern __constant__ Material c_materials[256];

int main() {
    Timer timer;
    
    // Define list of objects here
    std::vector<std::string> obj_list = {
        "light",      "donkey_kong", "left_wall", "back_wall",
        "right_wall", "ceiling",     "floor"};
    int list_size = obj_list.size();

    // Define colors here
    float3 white = make_float3(0.73f, 0.73f, 0.73f);
    float3 red = make_float3(0.65f, 0.05f, 0.05f);
    float3 green = make_float3(0.12f, 0.45f, 0.15f);
    std::vector<float3> albedo_list = {
        make_float3(0.0f, 0.0f, 0.0f), make_float3(102.0f/255.0f, 54.0f/255.0f, 5.0f/255.0f),
        red, white, green, white, white
    };

    Camera camera(cornell_box_params);
    SceneManager scene(camera, list_size);

    for (int i = 0; i < list_size; i++) {
        scene.addTriangleMeshColor(std::string(PROJECT_ROOT) +
                                  "/assets/cornell_box/" + obj_list[i] + ".obj",
                              albedo_list[i], i);
    }

    // Copy constant material to c_materials
    // CUDA_CHECK(cudaMemcpyToSymbol(c_materials, scene.materials,
    //     sizeof(Material) * list_size));

    scene.renderMesh();

    scene.copyFromDevice();
    scene.saveImage("cornell_box_test.ppm");

    return 0;
}
