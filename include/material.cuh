#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <cuda_runtime.h>

struct Material {
    float3 albedo; // RGB color
    // NOTE: add padding for alignment?? 
    // float padding; // 16 bytes
}

__constant__ Material c_materials[256];
#endif // MATERIAL_CUH