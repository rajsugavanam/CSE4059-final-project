#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <cuda_runtime.h>

struct Material {
    float3 albedo; // RGB color
    // NOTE: add padding for alignment?? 
    // float padding; // 16 bytes
};

#endif // MATERIAL_CUH