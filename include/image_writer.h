#ifndef IMAGE_WRITER_H
#define IMAGE_WRITER_H

#include <fstream>
#include "vec3.cuh"

void writeToPPM(const char* filename, Vec3* image_buffer, int pixel_width,
  int pixel_height) {
std::ofstream os(filename);
os << "P3\n" << pixel_width << " " << pixel_height << "\n255\n";
for (int j = 0; j < pixel_height; j++) {
for (int i = 0; i < pixel_width; i++) {
int pixel_idx = (j * pixel_width + i);
int r = static_cast<int>(image_buffer[pixel_idx].x() * 255.999);
int g = static_cast<int>(image_buffer[pixel_idx].y() * 255.999);
int b = static_cast<int>(image_buffer[pixel_idx].z() * 255.999);
os << r << " " << g << " " << b << "\n";
}
}
os.close();
}

#endif // IMAGE_WRITER_H