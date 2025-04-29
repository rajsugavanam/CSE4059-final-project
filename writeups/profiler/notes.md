# NOTES

# Reduction 

**[dk.obj](assets/dk.obj) AABB**: 4745 Triangles

| Implementation | x min | y min | z min | x max | y max | z max | Time |
|----------------|-------|-------|-------|-------|-------|-------|------|
| GPU::Triangle3 | -14.0306 | -19.0771 | -37.4049 | 10.0022 | 1.00416 | -20.444 | 337490 ns |
| CPU::Triangle3 | -14.0306 | -19.0771 | -37.4049 | 10.0022 | 1.00416 | -20.444 | 419135 ns |

# References

Small list of refs that might not be cited in the source code.

## CUDA

- [Building Cross-Platform CUDA Applications with CMake](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)
- [Separate Compilation and Linking of CUDA C++ Device Code](https://developer.nvidia.com/blog/separate-compilation-linking-cuda-device-code/#caveats)
- [Device Link Time Optimzation blog](https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/)
- [CMake DLTO for CUDA](https://cmake.org/cmake/help/latest/release/3.25.html#languages)

## Raytracing/Rendering

- [Raytracing in One Weekend series](https://raytracing.github.io/)
- [Physically Based Redering: From Theory to Implementation (Online Book)](https://www.pbrt.org/)
- [Cornell Box Data](https://www.graphics.cornell.edu/online/box/data.html)
- [LearnOpenGL](https://learnopengl.com/)

### Colorimetry

- [CIE Fundamentals for Color Measurements Yoshi Ohno](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=841491)
- [CIE XYZ Color Matching Functions paper](https://research.nvidia.com/publication/2013-07_simple-analytic-approximations-cie-xyz-color-matching-functions)
- [IEC Standard for XYZ to sRGB ](https://webstore.iec.ch/en/publication/6168)
- [sRGB Transfer Function (Gamma)](https://en.wikipedia.org/wiki/SRGB#Definition)
- [OpenGL HDR Tone Mapping](https://learnopengl.com/Advanced-Lighting/HDR) for simple exposure tone mapping

## Extra

- [Super Smash Bros. Melee: Donkey Kong Trophy `.obj` file](https://www.models-resource.com/gamecube/ssbm/model/38652/)
- [LHTSS sRGB to reflectance curves website/paper](http://scottburns.us/reflectance-curves-from-srgb-10/)
- [Spectral Cornell Box Shader Toy](https://www.shadertoy.com/view/WtlSWM)
- [Hero Wavelength Spectral Sampling](https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14_files/WNDWH14HWSS.pdf) A. Wilkie et al.
- [Spectral Raytracing Presentation slides Omercan Yazici](https://graphics.cg.uni-saarland.de/courses/ris-2021/slides/Spectral%20Raytracing.pdf)
- [Standford CS348 Computer Graphics: Image Synthesis Techniques](http://www.graphics.stanford.edu/courses/cs348b-01/) and from there [P. Hanrahan, Monte Carlo path tracing, SIGGRAPH 2001 Course 29: Monte Carlo Ray Tracing. (pdf)](http://www.graphics.stanford.edu/courses/cs348b-01/course29.hanrahan.pdf)