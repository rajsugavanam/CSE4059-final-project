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
| `dk_spectral_box` | 11892.626 s | Resolution: 1440x1440, 16384 spp |

% REDUCTION ARBITRARY LENGTH
% 5k data points
% CPU Min: 34806 ns * 9
% CPU Max: 59253 ns * 9
% GPU Min: 96714 ns
% GPU Max: 19877 ns

% 1 million data points
% CPU MIN: 7.368 ms
% CPU MAX: 7.039 ms
% GPU MIN: 155696 ns
% GPU MAX: 64262 ns

% 100 million
% CPU MIN: 714.632 ms
% CPU MAX: 711.017 ms
% GPU MIN: 4.705 ms
% GPU MAX: 4.617 ms

% REDUCTION: 4745 DK model
% CPU: 520167 ns
% GPU: 284420 ns with CUDA streams

% AABB SPEED UP DK_CB SCENE
% NO AABB: 93.963 ms
% AABB: 22.664 ms

| Render | 1spp | 4 spp | 16 spp | 64 spp | 256 spp | 1024 spp | 4096 spp | 16384 spp |
|--------|-------|--------|---------|---------|----------|----------|----------|-----------|
| DK     | 0.802676 ms | 3.003 s | 11.652 s | 46.893 s | 186.166 s | 743.864 s | 2982.163 s | 11892.626 s |
| CB     | 0.102828 ms | 0.201268 ms |0.801685 ms | 2.903 s | 11.809 s | 47.732 s | 186.330 s | 736.517 s |
| PBRT   | 0.5 s | 2 s | 7.6 s | 32.5 s | 181.6 s | - | - | - |

All tests use 5 max bounces, and 1440x1440 output resolution.
- DK Model: 4745 Triangle mesh with 10 triangle enclosing box and 2 triangle light source
- Cornell Box: 10 triangle walls/floor/ceiling, two 12 triangle small and large box, and
2 triangle light source
- PBRT: SimplePathIntegrator with Independent Sampler
