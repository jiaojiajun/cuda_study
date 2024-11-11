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

void verify_result(vector<float> a, vector<float> b, vector<float> c, int N){
    for (int i=0;i < N; i++){
        assert(a[i]+b[i] == c[i]);
    }
}
int main(){
    const int N = 1 << 16;
    const size_t nBytes = N * sizeof(float);

    vector<float> a;
    vector<float> b;
    vector<float> c;
    a.reserve(N);
    b.reserve(N);
    c.resize(N);


    for (int i=0; i < N; i++){
        a.push_back((float)(rand()%100));
        b.push_back((float)(rand()%100));
    }
    printf("succeed in initialize a & b\n");

    float* fpDeviceA, *fpDeviceB, *fpDeviceC;
    cudaMalloc(&fpDeviceA, nBytes);
    cudaMalloc(&fpDeviceB, nBytes);
    cudaMalloc(&fpDeviceC, nBytes);

    cudaMemcpy(fpDeviceA, a.data(), nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDeviceB, b.data(), nBytes, cudaMemcpyHostToDevice);
    
    int numThreads = 256;
    int numBlocks = (N + numThreads -1) /numThreads;

    vector_add_cuda<<<numBlocks, numThreads>>>(fpDeviceA, fpDeviceB, fpDeviceC, N);

    cudaMemcpy(c.data(),fpDeviceC, nBytes, cudaMemcpyDeviceToHost);
    
    verify_result(a,b,c,N);

    printf("compute successfully\n");
    
    return 0;
}

