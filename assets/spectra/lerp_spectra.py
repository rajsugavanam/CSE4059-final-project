import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee'])
black = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
red = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][2]
green = plt.rcParams['axes.prop_cycle'].by_key()['color'][3]
# print default dpi
# print(f"Default dpi: {plt.rcParams['figure.dpi']}")
# set dpi
# plt.rcParams['figure.dpi'] = 300

# Read the input CSV file with Cornell Box surface reflectances
input_path = 'surface_reflectances.csv'
output_path = 'surface_reflectances_1nm.csv'

# Load the data
df = pd.read_csv(input_path)

# Get min and max wavelengths from the data
min_wavelength = df['Wavelength'].min()  # 400nm
max_wavelength = df['Wavelength'].max()  # 700nm

# Create new wavelength range with 1nm step size
new_wavelengths = np.arange(min_wavelength, max_wavelength + 1, 1)

# Create new dataframe for interpolated values
new_df = pd.DataFrame({'Wavelength': new_wavelengths})

# Interpolate each color component
for color in ['white', 'green', 'red']:
    new_df[color] = np.interp(new_wavelengths, df['Wavelength'], df[color])

# Save to new CSV file
new_df.to_csv(output_path, index=False)

# Create C/C++ header array data
with open('../../include/util/cb_spectrum.h', 'w') as f:
    f.write('#ifndef CB_SPECTRUM_H\n')
    f.write('#define CB_SPECTRUM_H\n\n')
    f.write('// Cornell Box surface reflectances interpolated to 1nm\n')
    f.write('// Range: 400nm - 700nm, lerp 1nm step size => 301 Elements\n')
    f.write('// https://www.graphics.cornell.edu/online/box/data.html\n')
    
    # Write white reflectance
    f.write('const float WHITE_REFLECTANCE_SPECTRUM[301] = {\n    // White reflectance\n    ')
    values_per_line = 6
    for i, val in enumerate(new_df['white']):
        f.write(f"{val:.9f}f")
        if i < len(new_df['white']) - 1:
            f.write(', ')
        if (i + 1) % values_per_line == 0 and i != len(new_df['white']) - 1:
            f.write('\n    ')
    f.write('\n};\n\n')
    
    # Write green reflectance
    f.write('const float GREEN_REFLECTANCE_SPECTRUM[301] = {\n    // Green reflectance\n    ')
    for i, val in enumerate(new_df['green']):
        f.write(f"{val:.9f}f")
        if i < len(new_df['green']) - 1:
            f.write(', ')
        if (i + 1) % values_per_line == 0 and i != len(new_df['green']) - 1:
            f.write('\n    ')
    f.write('\n};\n\n')
    
    # Write red reflectance
    f.write('const float RED_REFLECTANCE_SPECTRUM[301] = {\n    // Red reflectance\n    ')
    for i, val in enumerate(new_df['red']):
        f.write(f"{val:.9f}f")
        if i < len(new_df['red']) - 1:
            f.write(', ')
        if (i + 1) % values_per_line == 0 and i != len(new_df['red']) - 1:
            f.write('\n    ')
    f.write('\n};\n')
    f.write('#endif // CB_SPECTRUM_H')

# Plot Interpolated Reflectance Spectra
plt.figure()
# plt.plot(df['Wavelength'], df['white'], 'o', label='White (Original)', markersize=1)
# plt.plot(df['Wavelength'], df['green'], 'o', label='Green (Original)', markersize=1, color=green)
# plt.plot(df['Wavelength'], df['red'], 'o', label='Red (Original)', markersize=1, color=red)
plt.plot(new_df['Wavelength'], new_df['white'], label='White', color=black)
plt.plot(new_df['Wavelength'], new_df['green'], label='Green', color=green)
plt.plot(new_df['Wavelength'], new_df['red'], label='Red', color=red)
plt.grid(True)
plt.ylim(0, 1)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Cornell Box Surface Reflectances - 1nm Interpolation')
plt.legend(frameon=True, loc='upper right')
plt.savefig('reflectance_interpolation.png')
print(f"Interpolated data saved to {output_path}")
print(f"C++ arrays saved to include/cb_spectrum.h")
print(f"Visualization saved to spectra/reflectance_interpolation.png")

# Define the light source emission spectrum data
light_data = {
    'Wavelength': [400.0, 500.0, 600.0, 700.0],
    'emission': [0.0, 8.0, 15.6, 18.4]
}

# Create DataFrame
light_df = pd.DataFrame(light_data)

# Save raw data to CSV for reference
light_df.to_csv('light_emission.csv', index=False)

# Create interpolated wavelength range (400nm to 700nm with 1nm steps)
new_wavelengths = np.arange(400, 701, 1)

# Interpolate the emission values
interpolated_emission = np.interp(new_wavelengths, light_df['Wavelength'], light_df['emission'])

# Create new DataFrame with interpolated values
new_light_df = pd.DataFrame({
    'Wavelength': new_wavelengths,
    'emission': interpolated_emission
})

# Save interpolated data to CSV
new_light_df.to_csv('light_emission_1nm.csv', index=False)

# Generate C++ array format for the emission spectrum
with open('../../include/util/cb_light_spectrum.h', 'w') as f:
    f.write('#ifndef CB_LIGHT_SPECTRUM_H\n')
    f.write('#define CB_LIGHT_SPECTRUM_H\n\n')
    f.write('// Light source emission spectrum interpolated to 1nm\n')
    f.write('// Range: 400nm - 700nm, lerp 1nm step size => 301 Elements\n')
    f.write('// https://www.graphics.cornell.edu/online/box/data.html\n')
    
    # Write light emission spectrum
    f.write('const float LIGHT_EMISSION_SPECTRUM[301] = {\n    // Light emission\n    ')
    values_per_line = 6
    for i, val in enumerate(new_light_df['emission']):
        f.write(f"{val:.9f}f")
        if i < len(new_light_df['emission']) - 1:
            f.write(', ')
        if (i + 1) % values_per_line == 0 and i != len(new_light_df['emission']) - 1:
            f.write('\n    ')
    f.write('\n};\n\n')
    
    # Also add the light reflectance (constant 0.78)
    f.write('const float LIGHT_REFLECTANCE_SPECTRUM[301] = {\n    // Light reflectance (constant 0.78)\n    ')
    for i in range(301):
        f.write("0.780000000f")
        if i < 300:
            f.write(', ')
        if (i + 1) % values_per_line == 0 and i != 300:
            f.write('\n    ')
    f.write('\n};\n')

    f.write('#endif // CB_LIGHT_SPECTRUM_H')

# Plot to visualize the interpolation results
plt.figure()
plt.plot(new_light_df['Wavelength'], new_light_df['emission'], label='Interpolated (1nm)')
plt.plot(light_df['Wavelength'], light_df['emission'], 'o', label='Original Data Points', markersize=3, color=black, alpha=0.5)
plt.grid(True)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Emission Intensity')
plt.title('Light Source Emission Spectrum Interpolation')
plt.legend(frameon=True)
plt.savefig('light_emission_interpolation.png')

print("Light emission data interpolated and saved")
print("C++ arrays saved to include/cb_light_spectrum.h")
print("Visualization saved to light_emission_interpolation.png")

# Plotting CIE_xyz_1931_2deg.csv data points
cie_data = np.genfromtxt('CIE_xyz_1931_2deg.csv', delimiter=',', skip_header=1)
cie_wavelengths = cie_data[:, 0]
cie_x = cie_data[:, 1]
cie_y = cie_data[:, 2]
cie_z = cie_data[:, 3]
plt.figure()
plt.plot(cie_wavelengths, cie_x, color=red, label=r'CIE x: $\bar{x}(\lambda)$')
plt.plot(cie_wavelengths, cie_y, color=green, label=r'CIE y: $\bar{y}(\lambda)$')
plt.plot(cie_wavelengths, cie_z, color=blue, label=r'CIE z: $\bar{z}(\lambda)$')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Tristimulus Values')
plt.title('CIE 1931 $XYZ$ Color Matching Functions')
plt.legend(frameon=True)
plt.grid(True)
plt.savefig('cie_color_matching_functions.png')
print("CIE color matching functions plotted and saved to cie_color_matching_functions.png")

# Save the CIE data to a C++ header file with sRBG matrix 

with open('../../include/util/cie_spectrum.h', 'w') as f:
    f.write('#ifndef CIE_SPECTRUM_H\n')
    f.write('#define CIE_SPECTRUM_H\n\n')
    f.write('// CIE XYZ to sRGB matrix\n')
    f.write('// https://stackoverflow.com/q/66360637\n')
    f.write('const float3 CIE_XYZ_TO_SRGB[3] = {\n')
    f.write('    make_float3(3.2404542f, -1.5371385f, -0.4985314f),\n')
    f.write('    make_float3(-0.9692660f, 1.8760108f,  0.0415560f),\n')
    f.write('    make_float3(0.0556434f, -0.2040259f,  1.0572252f)\n')
    f.write('};\n\n')
    
    # Write CIE color matching functions float3 array
    f.write('// CIE color matching functions\n')
    f.write('const float3 CIE_COLOR_MATCHING_FUNCTIONS[301] = {\n')
    for i in range(301):
        wavelength = 400 + i  # Start at 400nm and increment by 1nm
        index = np.where(cie_wavelengths == wavelength)[0][0]
        f.write(f"    make_float3({cie_x[index]:.9f}f, {cie_y[index]:.9f}f, {cie_z[index]:.9f}f)")
        if i < 300:
            f.write(',\n')

    f.write('\n};\n\n')
    f.write('#endif // CIE_SPECTRUM_H')

# WARN: I tried rewriting the matlab code but i was getting singular matrix errors...
# We can just manually copy rgbToXYZ for a given color via matlab or excel provided by
# http://scottburns.us/fast-rgb-to-spectrum-conversion-for-reflectances/
# Using fast Inverse Hyperbolic Tangent
# http://scottburns.us/reflectance-curves-from-srgb-10/