#include<stdio.h>
#include"./common.cuh"

int main(){
    float* fpHost;
    size_t nBytes = 1024 * sizeof(float);
    fpHost = (float*)malloc(nBytes);

    float* fpDevice;
    cudaError_t error = cudaMalloc(&fpDevice, nBytes);
    errorCheck(cudaMemcpy(fpDevice, fpHost, nBytes, cudaMemcpyDeviceToHost),__FILE__, __LINE__);
}