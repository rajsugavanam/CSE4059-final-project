#include <iostream>
#include <math/VecN.cuh>
#include <ostream>

#define TEST_SIZE 4

int main() {

    float vec1arr[TEST_SIZE] = {1.0, 2.0, 3.0, 4.0};
    float vec2arr[TEST_SIZE] = {7.0, 3.0, -1.0, 8.0};

    VecN<float>* floatVec1 = new VecN<float>(TEST_SIZE, vec1arr);
    VecN<float>* floatVec2 = new VecN<float>(TEST_SIZE, vec2arr);
    VecN<float>* resultVec = floatVec1->deviceAdd(floatVec2);

    int N = resultVec->N;
    float* contents = resultVec->pv;
    for (int i=0; i<N; i++) {
        std::cout << i << ": " << contents[i] << std::endl;
    }

    return 0;
}
