#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

// CUDA Error Handling
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(error) << std::endl;                 \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

#endif // CUDA_HELPER_H