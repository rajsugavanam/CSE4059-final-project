#include "camera.h"
#include "scene_manager.cuh"
#include "timer.h"

int main() {
    Timer timer;
    Camera camera(cornell_box_params);
    SceneManager scene(camera, 3);

    scene.addTriangleMesh(std::string(PROJECT_ROOT) + "/assets/cornell_box/light.obj", 0);
    scene.addTriangleMesh(std::string(PROJECT_ROOT) + "/assets/cornell_box/donkey_kong.obj", 1);
    scene.addTriangleMesh(std::string(PROJECT_ROOT) + "/assets/cornell_box/cornell_box.obj", 2);

    scene.renderMesh();

    scene.copyFromDevice();
    scene.saveImage("cornell_box_test.ppm");

    return 0;
}
