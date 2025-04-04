# CUDA RAY TRACER (CRT)

A physically-based rendering system ray tracer that takes advantage of the CUDA API.

## Building
Invoke `cmake --build .` from the debug directory to build an executable with debug symbols. Invoke from release to exclude them. If build issues arise on your system, try invoking `cmake ..` from the directories, or editing `CMakeCache.txt` to ensure the correct architecture, etc is used.
