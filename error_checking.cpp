#include <hip/hip_runtime.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#define NUM 1024

int main(void) {
    float *gpuA = 0;

    HIP_ASSERT(hipMalloc((void**)&gpuA, NUM * sizeof(float)));

    return 0;
}