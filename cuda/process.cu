#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void addArray(int *ary1, int *ary2)
{
    int indx = threadIdx.x;
    ary1[indx] = ary2[indx];
}

