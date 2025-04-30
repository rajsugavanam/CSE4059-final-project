import numpy as np
import os
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])
black = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
red = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
green = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]

# Samples per pixel values
spp_values = [1, 4, 16, 64, 256, 1024, 4096, 16384]

# Render times (in seconds)
# DK model render times
dk_render_times = [
    0.802676,         # 1 spp: 802.676 ms
    3.003,            # 4 spp: 3.003 s
    11.652,           # 16 spp: 11.652 s
    46.893,           # 64 spp: 46.893 s
    186.166,          # 256 spp: 186.166 s
    743.864,          # 1024 spp: 743.864 s
    2982.163,         # 4096 spp: 2982.163 s
    11892.626         # 16384 spp: 11892.626 s
]

# Cornell Box render times (convert ms to s where needed)
cb_render_times = [
    0.102828,         # 1 spp: 102.828 ms
    0.201268,         # 4 spp: 201.268 ms
    0.801685,         # 16 spp: 801.685 ms
    2.903,            # 64 spp: 2.903 s
    11.809,           # 256 spp: 11.809 s
    47.732,           # 1024 spp: 47.732 s
    186.330,          # 4096 spp: 186.330 s
    736.517           # 16384 spp: 736.517 s
]

# PBRT render times
pbrt_render_times = [
    0.5,              # 1 spp: 0.5 s
    2.0,              # 4 spp: 2 s
    7.6,              # 16 spp: 7.6 s
    32.5,             # 64 spp: 32.5 s
    181.6,            # 256 spp: 181.6 s
]

# Calculate average speed up between Cornell box and pbrt
# Speed up = PBRT time / Cornell Box time
pbrt_speedup = [
    pbrt_render_times[0] / cb_render_times[0],  # 1 spp
    pbrt_render_times[1] / cb_render_times[1],  # 4 spp
    pbrt_render_times[2] / cb_render_times[2],  # 16 spp
    pbrt_render_times[3] / cb_render_times[3],  # 64 spp
    pbrt_render_times[4] / cb_render_times[4],  # 256 spp
]
# Print speed up values
for i, speedup in enumerate(pbrt_speedup):
    print(f"Speed up at {spp_values[i]} SPP: {speedup:.2f}x")
# calculate average speed up
average_speedup = np.mean(pbrt_speedup)
print(f"Average speed up: {average_speedup:.2f}x")

# Extrapolate PBRT render times for higher SPP values
# Calculate the ratio between 256 SPP and 64 SPP to determine scaling factor
pbrt_scaling_factor = pbrt_render_times[4] / pbrt_render_times[3]  # 181.6 / 32.5
pbrt_extrapolated_times = pbrt_render_times.copy()

# Extrapolate for 1024 SPP
pbrt_extrapolated_times.append(pbrt_render_times[4] * 4)  # 181.6 * 4 = ~726.4 s

# Extrapolate for 4096 SPP
pbrt_extrapolated_times.append(pbrt_extrapolated_times[5] * 4)  # 726.4 * 4 = ~2905.6 s

# Extrapolate for 16384 SPP
pbrt_extrapolated_times.append(pbrt_extrapolated_times[6] * 4)  # 2905.6 * 4 = ~11622.4 s

# Plot data with symlog scale (linear for small values, log for large values)
plt.figure(figsize=(5, 3))

# Use symlog scale for y-axis (linear below linthresh, log above)
plt.xscale('log')  # Keep x-axis as log scale
plt.yscale('log')  # Linear below 10 seconds, log above

# Plot the data
plt.plot(spp_values, dk_render_times, 'o-', label='CRT: DK Model', color=black)
plt.plot(spp_values, cb_render_times, 's-', label='CRT: Cornell Box', color=red)
plt.plot(spp_values[:5], pbrt_render_times, '^-', label='PBRT: Cornell Box', color=blue)

plt.xlabel('Samples Per Pixel (SPP)')
plt.ylabel('Render Time (seconds)')
plt.title('Samples Per Pixel vs Rendering Time')
plt.grid(True)
plt.legend(frameon=True)
plt.tight_layout()

# Save the figure
# plt.savefig('render_time_vs_spp.pdf', dpi=300, bbox_inches='tight')
plt.savefig('render_time_vs_spp.png')
# plt.show()

# Plot data without log scale
plt.figure(figsize=(5, 3))
plt.plot(spp_values, dk_render_times, 'o-', label='CRT: DK Model', color=black)
plt.plot(spp_values, cb_render_times, 's-', label='CRT: Cornell Box', color=red)
plt.plot(spp_values[:5], pbrt_render_times, '^-', label='PBRT: Cornell Box (Measured)', color=blue)
plt.plot(spp_values[5:], pbrt_extrapolated_times[5:], '^-', label='PBRT: Cornell Box (Extrapolated)', color=blue, linestyle='--')
plt.xlabel('Samples Per Pixel (SPP)')
plt.ylabel('Render Time (seconds)')
plt.title('Samples Per Pixel vs Rendering Time')
plt.grid(True)
plt.legend(frameon=True)
plt.tight_layout()
# Save the figure
plt.savefig('render_time_vs_spp_nonlog.png')