#include<stdio.h>

int main(){
    float* fpDevice;
    size_t nBytes = 4;
    cudaError_t error = cudaMalloc(&fpDevice, nBytes);
    cudaMemset(fpDevice, 0, nBytes);
    
    float* fpHost;
    fpHost = (float*)malloc(nBytes);
    cudaMemcpy(fpHost, fpDevice, nBytes, cudaMemcpyDeviceToHost);

    printf("the number is %.2f", *fpHost);
    return 0;
}