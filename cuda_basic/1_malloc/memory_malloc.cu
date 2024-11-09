#include<stdio.h>

int main(){
    // memory malloc in host
    float * fpHost = NULL;
    size_t nBytes = 1024;
    fpHost = (float*) malloc(nBytes);
    if (fpHost==NULL){
        printf("failed to malloc host memory\n");
        exit(-1);
    }

    float* fpDevice;
    cudaError_t error = cudaMalloc(&fpDevice,nBytes);
    if (error!= cudaSuccess){
        printf("failed to malloc device memory\n");
        exit(-1);
    }
    printf("both the host and device has been malloced memory\n");
    return 0;
}