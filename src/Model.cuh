#ifndef MODEL_CUH
#define MODEL_CUH

#include <vector>
#include "triangle3.cuh"

class Face {
    public:
        const Triangle3 triFace;
        const Norm3 vertexNormals;

        Face(const Triangle3 triFace, const Norm3 vertexNormals);
};

class Model {
    public:
        std::vector<Face> modelFaces;
        // std::vector<Vec3> modelVertices;
        // std::vector<Vec3> modelNormals;

    Model();
};

#endif
