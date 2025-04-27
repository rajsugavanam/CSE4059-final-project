#ifndef SPECTRUM_CUH
#define SPECTRUM_CUH

#include <cuda_runtime.h>

#include "cb_light_spectrum.h"
#include "cb_spectrum.h"
#include "cie_spectrum.h"

// Declare constants as external (they're defined in scene_manager.cu)
extern __constant__ float3 c_cieXYZ[301];
extern __constant__ float3 c_cieXYZ_to_sRGB[3];
extern __constant__ float c_white_reflectance[301];
extern __constant__ float c_green_reflectance[301];
extern __constant__ float c_red_reflectance[301];
extern __constant__ float c_light_emission[301];
extern __constant__ float c_light_reflectance[301];

// Wavelength constants
constexpr float MIN_WAVELENGTH = 400.0f;
constexpr float MAX_WAVELENGTH = 700.0f;
constexpr int SPECTRUM_SAMPLES = 301;

// SampledWavelength class
class SampledWavelength {
   public:
    float lambda;  // The sampled wavelength in nm
    int idx;       // Index in the spectrum array

    // Constructor
    __host__ __device__ SampledWavelength(float wavelength)
        : lambda(wavelength) {
        // Clamp wavelength to valid range
        lambda = fmaxf(MIN_WAVELENGTH, fminf(MAX_WAVELENGTH, lambda));

        // Calculate array index
        idx = static_cast<int>(lambda - MIN_WAVELENGTH);
    }

    // Sample CIE XYZ for this wavelength
    __device__ float3 sampleCIE() const {
        return make_float3(c_cieXYZ[idx].x, c_cieXYZ[idx].y, c_cieXYZ[idx].z);
    }
};

// SampledSpectrum class
class SampledSpectrum {
   private:
    float values[SPECTRUM_SAMPLES];

   public:
    // Default constructor
    __host__ __device__ SampledSpectrum() {
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            values[i] = 0.0f;
        }
    }

    // Constructor from constant array pointer
    __host__ __device__ SampledSpectrum(const float* spectrum) {
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            values[i] = spectrum[i];
        }
    }

    // Sample the spectrum at a specific wavelength
    __device__ float sample(const SampledWavelength& wl) const {
        return values[wl.idx];
    }

    // Convert spectrum to XYZ
    __device__ float3 toXYZ() const {
        float3 xyz = make_float3(0.0f, 0.0f, 0.0f);

        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            xyz.x += values[i] * c_cieXYZ[i].x;
            xyz.y += values[i] * c_cieXYZ[i].y;
            xyz.z += values[i] * c_cieXYZ[i].z;
        }

        // Normalize by number of samples
        float scale = (MAX_WAVELENGTH - MIN_WAVELENGTH) / SPECTRUM_SAMPLES;
        return make_float3(xyz.x * scale, xyz.y * scale, xyz.z * scale);
    }

    // WARN: This isn't Hero wavelength sampling method
    // It samples a wavelength based on a uniform distribution
    // Hero wavelength samples multiple wavelengths offset from the first
    // sampled wavelength per ray.
    __device__ static SampledWavelength SampleWavelength(float u) {
        // Map u [0,1] to wavelength range [MIN_WAVELENGTH, MAX_WAVELENGTH]
        float lambda = MIN_WAVELENGTH + u * (MAX_WAVELENGTH - MIN_WAVELENGTH);
        return SampledWavelength(lambda);
    }

    // Create a single-wavelength spectrum for a sampled wavelength
    __device__ static SampledSpectrum SingleWavelength(
        const SampledWavelength& wl) {
        SampledSpectrum result;
        // Set all values to zero except at the sampled wavelength
        result.values[wl.idx] = 1.0f;
        return result;
    }

    // WARN: Unused
    // Set value at specific wavelength
    __device__ void setValue(const SampledWavelength& wl, float value) {
        values[wl.idx] = value;
    }

    // Static methods to create common spectra
    __device__ static SampledSpectrum WhiteReflectance() {
        return SampledSpectrum(c_white_reflectance);
    }

    __device__ static SampledSpectrum GreenReflectance() {
        return SampledSpectrum(c_green_reflectance);
    }

    __device__ static SampledSpectrum RedReflectance() {
        return SampledSpectrum(c_red_reflectance);
    }

    __device__ static SampledSpectrum LightEmission() {
        return SampledSpectrum(c_light_emission);
    }

    __device__ static SampledSpectrum LightReflectance() {
        return SampledSpectrum(c_light_reflectance);
    }

    // Convert to RGB (through XYZ)
    __device__ float3 toRGB() const {
        float3 xyz = toXYZ();

        // Apply XYZ to RGB transformation
        float r = c_cieXYZ_to_sRGB[0].x * xyz.x +
                  c_cieXYZ_to_sRGB[0].y * xyz.y + c_cieXYZ_to_sRGB[0].z * xyz.z;
        float g = c_cieXYZ_to_sRGB[1].x * xyz.x +
                  c_cieXYZ_to_sRGB[1].y * xyz.y + c_cieXYZ_to_sRGB[1].z * xyz.z;
        float b = c_cieXYZ_to_sRGB[2].x * xyz.x +
                  c_cieXYZ_to_sRGB[2].y * xyz.y + c_cieXYZ_to_sRGB[2].z * xyz.z;

        return make_float3(r, g, b);
    }

    // Arithmetic operations
    __device__ SampledSpectrum operator*(const SampledSpectrum& other) const {
        SampledSpectrum result;
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            result.values[i] = values[i] * other.values[i];
        }
        return result;
    }

    __device__ SampledSpectrum operator*(float scalar) const {
        SampledSpectrum result;
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            result.values[i] = values[i] * scalar;
        }
        return result;
    }

    __device__ SampledSpectrum& operator+=(const SampledSpectrum& other) {
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            values[i] += other.values[i];
        }
        return *this;
    }

    __device__ SampledSpectrum operator+(const SampledSpectrum& other) const {
        SampledSpectrum result;
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            result.values[i] = values[i] + other.values[i];
        }
        return result;
    }

    __device__ SampledSpectrum& operator*=(float scalar) {
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            values[i] *= scalar;
        }
        return *this;
    }

    // WARN: Unused, might be useful for exiting early
    __device__ bool isBlack() const {
        for (int i = 0; i < SPECTRUM_SAMPLES; ++i) {
            if (values[i] > 0.0f) return false;
        }
        return true;
    }
};

#endif  // SPECTRUM_CUH