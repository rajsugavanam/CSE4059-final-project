#ifndef MODEL_CUH
#define MODEL_CUH

#include <vector>
#include "triangle3.cuh"

class Model {
    public:
        std::vector<Triangle3> modelTriangles;

        Model() : modelTriangles(std::vector<Triangle3>()) {}
};

#endif
