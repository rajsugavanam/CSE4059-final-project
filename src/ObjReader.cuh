#ifndef OBJREADER_CUH
#define OBJREADER_CUH

#include <fstream>
#include <sstream>
#include <string>
#include "Model.cuh"

class ObjReader {
    private:
        std::string fileName;
    public:
        Model parsedModel;

        ObjReader(const std::string& fileName) : fileName(fileName) {}
        // __host__ void readModel();
        __host__ void readModel() {
            std::ifstream file_is(fileName, std::ios::in); // open to read.

            // read all lines of file
            std::vector<Vec3> modelVertices;
            std::vector<Vec3> modelNormals;

            std::string currLine;
            while( std::getline(file_is, currLine) ) {
                std::istringstream iss(currLine);
                std::string word;
                if (!(iss >> word)) {
                    throw std::runtime_error("[ERROR, ObjReader::readModel()]: Invalid .obj file!");
                }
                // VERTEX
                if (word == "v") {
                    float vecX;
                    float vecY;
                    float vecZ;

                    iss >> word;
                    vecX = std::stof(word);
                    iss >> word;
                    vecY = std::stof(word);
                    iss >> word;
                    vecZ = std::stof(word);

                    modelVertices.push_back(Vec3(vecX, vecY, vecZ));

                    // space_idx = feature_data.find(' '); // x[ ]y z
                    // model_feature = feature_data.substr(0, space_idx); // x
                    // feature_data = feature_data.substr(space_idx+1); // y z
                    // vec_x = std::stof(model_feature);

                    // space_idx = feature_data.find(' '); // y[ ]z
                    // model_feature = feature_data.substr(0, space_idx); // y
                    // feature_data = feature_data.substr(space_idx+1); // z
                    // vec_y = std::stof(model_feature);
                    // parsed_model.model_vertices.push_back()
                }
                // VERTEX NORMALS
                else if (word == "vn") {
                    float vecX;
                    float vecY;
                    float vecZ;

                    iss >> word;
                    vecX = std::stof(word);
                    iss >> word;
                    vecY = std::stof(word);
                    iss >> word;
                    vecZ = std::stof(word);

                    modelNormals.push_back(Vec3(vecX, vecY, vecZ));
                }
                // FACES
                else if (word == "f") {

                    const std::string delimiter = "/"; 

                    iss >> word;
                    // std::cout << word << std::endl;
                    std::string v1 = word.substr(0, word.find(delimiter));
                    word = word.substr(v1.length() + 1);
                    std::string vt1 = word.substr(0, word.find(delimiter));
                    word = word.substr(vt1.length() + 1);
                    std::string vn1 = word.substr(0, word.find(delimiter));

                    iss >> word;
                    // std::cout << word << std::endl;
                    std::string v2 = word.substr(0, word.find(delimiter));
                    word = word.substr(v2.length() + 1);
                    std::string vt2 = word.substr(0, word.find(delimiter));
                    word = word.substr(vt2.length() + 1);
                    std::string vn2 = word.substr(0, word.find(delimiter));

                    iss >> word;
                    // std::cout << word << std::endl;
                    std::string v3 = word.substr(0, word.find(delimiter));
                    word = word.substr(v3.length() + 1);
                    std::string vt3 = word.substr(0, word.find(delimiter));
                    word = word.substr(vt3.length() + 1);
                    std::string vn3 = word.substr(0, word.find(delimiter));

                    int v1Idx = std::stoi(v1)-1;
                    int v2Idx = std::stoi(v2)-1;
                    int v3Idx = std::stoi(v3)-1;

                    int vn1Idx = std::stoi(vn1)-1;
                    int vn2Idx = std::stoi(vn2)-1;
                    int vn3Idx = std::stoi(vn3)-1;

                    Triangle3 triangle = Triangle3(
                            modelVertices.at(v1Idx),
                            modelVertices.at(v2Idx),
                            modelVertices.at(v3Idx),
                            modelNormals.at(vn1Idx),
                            modelNormals.at(vn2Idx),
                            modelNormals.at(vn3Idx)
                            );
                    parsedModel.modelTriangles.push_back(triangle);
                }
            }
        }
};

#endif
