#pragma once

template <typename T>
class VecN {
    public:

        static const int BLOCK_SIZE = 512;
        int N;
        T* pv;

        VecN<T>(int N, T* pv) : N(N), pv(pv) {}
        VecN<T>(int N) : N(N), pv(new T[N]) {}
        ~VecN<T>();


        VecN<T>* deviceAdd(VecN<T>* vec2);


};

template <typename T>
__global__ void kerAdd(T* oper1, T* oper2, T* res, int N);
