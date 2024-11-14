#include<stdio.h>
#include<vector>
#include<cassert>
using namespace std;


__global__ void vector_add_cuda(float* fpA, float* fpB, float* fpC, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N){
        fpC[idx] = fpA[idx]+ fpB[idx];
    }
}

void verify_result(float* fpA, float* fpB, float* fpC, int N){
    for (int i=0;i < N; i++){
        assert(fpA[i]+fpB[i] == fpC[i]);
    }
}
int main(){
    const int N = 1 << 16;
    const size_t nBytes = N * sizeof(float);


    

    float* fpDeviceA, *fpDeviceB, *fpDeviceC;
    cudaMallocManaged(&fpDeviceA, nBytes);
    cudaMallocManaged(&fpDeviceB, nBytes);
    cudaMallocManaged(&fpDeviceC, nBytes);
    for (int i=0; i < N; i++){
        fpDeviceA[i] = ((float)(rand()%100));
        fpDeviceB[i] = ((float)(rand()%100));
    }

    
    int numThreads = 256;
    int numBlocks = (N + numThreads -1) /numThreads;

    vector_add_cuda<<<numBlocks, numThreads>>>(fpDeviceA, fpDeviceB, fpDeviceC, N);
    cudaDeviceSynchronize();
    
    verify_result(fpDeviceA, fpDeviceB, fpDeviceC,N);

    printf("compute successfully\n");
    
    return 0;
}

