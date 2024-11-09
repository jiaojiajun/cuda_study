#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

void initData(float* fp, int elementsNum){
    if (fp == NULL){
        printf("init data error, null pointer\n");
        return;
    }
    for (int i = 0; i < elementsNum; i++){
        fp[i] = (float)(rand() & 0xff) / 10.0f;
    }
}

__global__ void plusInGpu(float* A, float* B, float* C, int elementsNum){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elementsNum){
        C[i] = A[i] + B[i];
    }
}

int main(){
    int deviceCnt = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCnt);
    if (error != cudaSuccess || deviceCnt == 0){
        printf("can not find compatible device!\n");
        return -1;
    } else {
        printf("found device count: %d\n", deviceCnt);
    }

    int deviceId = 0;
    error = cudaSetDevice(deviceId);
    if (error != cudaSuccess){
        printf("can not set device 0 for computation\n");
        return -1;
    }

    int elementsNum = 2048;
    size_t nBytes = elementsNum * sizeof(float);

    float* fpHostA = (float*)malloc(nBytes);
    float* fpHostB = (float*)malloc(nBytes);
    float* fpHostC = (float*)malloc(nBytes);

    if (fpHostA == NULL || fpHostB == NULL || fpHostC == NULL){
        printf("failed to malloc host memory\n");
        return -1;
    }

    float* fpDeviceA = NULL;
    float* fpDeviceB = NULL;
    float* fpDeviceC = NULL;

    error = cudaMalloc(&fpDeviceA, nBytes);
    if (error != cudaSuccess){
        printf("can not malloc size %zu Bytes device memory for A\n", nBytes);
        return -1;
    }
    error = cudaMalloc(&fpDeviceB, nBytes);
    if (error != cudaSuccess){
        printf("can not malloc size %zu Bytes device memory for B\n", nBytes);
        cudaFree(fpDeviceA);
        return -1;
    }
    error = cudaMalloc(&fpDeviceC, nBytes);
    if (error != cudaSuccess){
        printf("can not malloc size %zu Bytes device memory for C\n", nBytes);
        cudaFree(fpDeviceA);
        cudaFree(fpDeviceB);
        return -1;
    }

    srand(123);
    initData(fpHostA, elementsNum);
    initData(fpHostB, elementsNum);

    error = cudaMemcpy(fpDeviceA, fpHostA, nBytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("failed to copy data from host to device\n");
        return -1;
    }
    error = cudaMemcpy(fpDeviceB, fpHostB, nBytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        printf("failed to copy data from host to device\n");
        return -1;
    }

    dim3 block(256);
    dim3 grid((elementsNum + block.x - 1) / block.x);
    plusInGpu<<<grid, block>>>(fpDeviceA, fpDeviceB, fpDeviceC, elementsNum);
    cudaDeviceSynchronize();

    error = cudaMemcpy(fpHostC, fpDeviceC, nBytes, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess){
        printf("failed to copy data from device to host\n");
        return -1;
    }

    for (int i = 0; i < 10; i++){
        printf("c[%d] is %f\t", i, fpHostC[i]);
    }

    free(fpHostA);
    free(fpHostB);
    free(fpHostC);

    cudaFree(fpDeviceA);
    cudaFree(fpDeviceB);
    cudaFree(fpDeviceC);

    return 0;
}