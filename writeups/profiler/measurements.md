# Measurements

This is a collection of measurements and benchmarks for the tests and main raytracer.

# Ctests

| Test | Time | Notes |
|------|------|-------|
| AABB.Render.Cube | 162738 ns | JS GPU: x5 Samples |

# Spectral Renderer

Samples per pixel (spp) are the number of rays cast per pixel. For $N$ samples per pixel,
the asymptotic runtime complexity is roughly $O(N)$.

| Test | Time | Notes |
| `cornell_box_spectral_test.cu` | 3022.609 s | Resolution: 1440x1440, 4096 spp | 