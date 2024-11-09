#include<stdio.h>

__global__ void add_vector_gpu(float* A, float* B, float* C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i]+ B[i];

}

int main(){
    int deviceCnt = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCnt);
    if (error != cudaSuccess || deviceCnt == 0){
        printf("found no compatible gpu!\n");
        exit(-1);
    }else{
        printf("found %d compatible gpu\n", deviceCnt);
    }
    int devId = 0;
    error = cudaSetDevice(devId);
    if(error!=cudaSuccess){
        printf("can't set device 0\n");
        exit(-1);
    }else {
        printf("successfully set cuda 0 as default device\n");
    }

    
    
}