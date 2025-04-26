#include <string>
#include "camera.h"
#include "scene_manager.cuh"
#include "timer.h"
#include <vector>

int main() {
    Timer timer;
    std::vector<std::string> obj_list = { "light", "donkey_kong", "left_wall", "back_wall", "right_wall", "ceiling", "floor" };
    int list_size = obj_list.size();
    Camera camera(cornell_box_params);
    SceneManager scene(camera, list_size); 

    // scene.addTriangleMesh(std::string(PROJECT_ROOT) + "/assets/cornell_box/light.obj", 0);
    // scene.addTriangleMesh(std::string(PROJECT_ROOT) + "/assets/cornell_box/donkey_kong.obj", 1);
    for (int i = 0; i < list_size; i++) {
        scene.addTriangleMesh(std::string(PROJECT_ROOT) + "/assets/cornell_box/" + obj_list[i] + ".obj", i);
    }
    scene.renderMesh();

    scene.copyFromDevice();
    scene.saveImage("cornell_box_test.ppm");

    return 0;
}
