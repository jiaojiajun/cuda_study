#include<stdio.h>
#include"./common.cuh"

__global__ void divideZero(float* fp){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    fp[i] = 0; 
}


int main(){
    float* fpDevice;
    size_t nBytes = 128 * sizeof(float);
    cudaMalloc(&fpDevice, nBytes);
    
    divideZero<<<1, 1025>>>(fpDevice); // max 1024 this is an error 

    errorCheck(cudaGetLastError(), __FILE__, __LINE__);
    errorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
    return 0;

}

