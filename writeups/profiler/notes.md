# NOTES

# Reduction 

**[dk.obj](assets/dk.obj) AABB**: 4745 Triangles

| Implementation | x min | y min | z min | x max | y max | z max | Time |
|----------------|-------|-------|-------|-------|-------|-------|------|
| GPU::Triangle3 | -14.0306 | -19.0771 | -37.4049 | 10.0022 | 1.00416 | -20.444 | 337490 ns |
| CPU::Triangle3 | -14.0306 | -19.0771 | -37.4049 | 10.0022 | 1.00416 | -20.444 | 419135 ns |