#include "helpers.h"

int main(int argc, char **argv)
{
    CUfunction kernel;
    init_kernel(argc, argv, &kernel);

    #define DTYPE float
    int count = 10;
    float *h_A, *h_B;
    CUdeviceptr d_A, d_B;
    int size_mem = init_mem<DTYPE>(count, &h_A, &d_A, InitMode::ZERO);
    init_mem<DTYPE>(count, &h_B, &d_B, InitMode::STEP);

    void *args[] = { &d_A, &d_B };
    checkCudaErrors(
        cuLaunchKernel(kernel, 
            count, 1, 1, // blockIdx x, y, z
            1, 1, 1, // threadIdx x, y, z
            0, // Shared mem bytes
            NULL, // hStream
            args, // Kernel params
            NULL // extra
        )
    );
    checkCudaErrors(cuMemcpyDtoH(h_A, d_A, size_mem));
    for (int i=0; i < count; i++) {
        printf("%f\n", h_A[i]);
    }
}
