#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_ASSERT(x) (assert(x == hipSuccess))

constexpr size_t N { 1000000 };
constexpr size_t vectorSizeBytes = N * sizeof(double_t);

__global__ void vectorAdd(double_t* A, double_t* B, double_t* C, size_t vectorSize) {
    size_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < vectorSize) {
        C[globalThreadIdx] = A[globalThreadIdx] + B[globalThreadIdx];
    }
}

int main(void) {
    double_t* CPUArrayA { (double_t*) malloc(vectorSizeBytes) };
    double_t* CPUArrayB { (double_t*) malloc(vectorSizeBytes) };
    double_t* CPUArrayC { (double_t*) malloc(vectorSizeBytes) };
    double_t* CPUVerifyArrayC { (double_t*) malloc(vectorSizeBytes) };

    for (size_t i = 0; i < N; ++i) {
        CPUArrayA[i] = i;
        CPUArrayB[i] = i;
    }

    double_t* GPUArrayA {};
    double_t* GPUArrayB {};
    double_t* GPUArrayC {};

    HIP_ASSERT(hipMalloc(&GPUArrayA, vectorSizeBytes));
    HIP_ASSERT(hipMalloc(&GPUArrayB, vectorSizeBytes));
    HIP_ASSERT(hipMalloc(&GPUArrayC, vectorSizeBytes));

    HIP_ASSERT(hipMemcpy(GPUArrayA, CPUArrayA, vectorSizeBytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(GPUArrayB, CPUArrayB, vectorSizeBytes, hipMemcpyHostToDevice));

    size_t blockSize { 1024 };
    size_t gridSize { (N + blockSize - 1) / blockSize };

    std::cout << "Threads per block: " << blockSize << "\n";
    std::cout << "Blocks per grid: " << gridSize << "\n";

    vectorAdd<<<gridSize, blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, N);
    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipMemcpy(CPUArrayC, GPUArrayC, vectorSizeBytes, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        CPUVerifyArrayC[i] = CPUArrayA[i] + CPUArrayB[i];
    }

    // I could use some small value to consider floating imprecision
    for (size_t i = 0; i < N; ++i) {
        if (CPUVerifyArrayC[i] - CPUArrayC[i] != 0.) {
            std::cout << "Wrong!\n";
        }
    }

    HIP_ASSERT(hipFree(GPUArrayA));
    HIP_ASSERT(hipFree(GPUArrayB));
    HIP_ASSERT(hipFree(GPUArrayC));

    free(CPUArrayA);
    free(CPUArrayB);
    free(CPUArrayC);
    free(CPUVerifyArrayC);

    return 0;
}