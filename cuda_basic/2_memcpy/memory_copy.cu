#include<stdio.h>

int main(){
    float * fpHostSrc;
    float * fpHostDest;
    float * fpDevice;

    size_t nBytes = 1024;
    fpHostSrc = (float*) malloc(nBytes);
    fpHostDest = (float*) malloc(nBytes);
    cudaError_t error = cudaMalloc(&fpDevice, nBytes);
    if (error != cudaSuccess){
        printf("malloc device memory failed\n");
    }

    memset(fpHostSrc, 0, nBytes);

    // memcpy(fpHostDest, fpHostSrc, nBytes); // 等价于下面这一行
    cudaMemcpy(fpHostDest, fpHostSrc, nBytes, cudaMemcpyHostToHost);
    printf("copy data from host to host succeed\n");

    cudaMemcpy(fpDevice, fpHostSrc, nBytes, cudaMemcpyHostToDevice);
    // 该函数一共有五种种复制类型
    // {host, device} x {host, device}  + default
    // 最后一种类型仅在支持统一虚拟寻址的系统上使用, 只能判断
    printf("copy data from host to device succeed\n");
    return 0;
}