#include <hip/hip_runtime.h>
#include <iostream>

__global__
void gpuHello() 
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello World from thread %d\n", threadId);
}

int main(void) {
    gpuHello<<<1,64>>>();
    hipError_t hip_error = hipDeviceSynchronize();

    std::cout << "Hip Error: " << hip_error << "\n";

    return 0;
}
