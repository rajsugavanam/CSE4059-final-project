#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <cuda_runtime.h>

enum class MaterialType {
    DIFFUSE = 0,
    SPECULAR = 1,
    EMISSIVE = 2
};

struct Material {
    float3 albedo;                 // RGB color
    int spectral_reflectance_id;   // ID for spectral reflectance
    int spectral_emission_id;      // ID for spectral emission
    bool is_emissive;              // 0 = false, 1 = true
    MaterialType type;             // Material type enumeration
};

#endif // MATERIAL_CUH