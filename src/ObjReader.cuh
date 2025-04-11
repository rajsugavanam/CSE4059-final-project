#include <string>
#include "Model.cuh"

class ObjReader {
    private:
        std::string fileName;
        Model parsedModel;
    public:
        ObjReader(const std::string& fileName);
        __host__ void readModel();
};
