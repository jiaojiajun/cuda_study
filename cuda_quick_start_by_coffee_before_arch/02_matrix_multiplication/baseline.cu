#include<stdio.h>
#include<cassert>
using namespace std;

__global__ void matrixMultiply(float* A, float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col;
    int tmp=0;
    for (int i=0; i<N; i++){
        tmp += A[row*N + i] * B[col*N+ i];
    }
    C[idx] = tmp;
}

void verifyResult(float* A, float* B, float* C, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float tmp = 0;
            for (int k =0; k < N; k++){
                tmp += A[i*N+k] * B[k+j*N];
            }
            // printf("c[i][j] is %f, while tmp is %f\n",C[i*N+j],tmp);
            assert(C[i*N +j] == tmp);
        }
        
    }
    
    
}

int main(){
    const int N = 1 << 10;
    const size_t nBytes = N*N * sizeof(float);

    float *fpHostA, *fpHostB, *fpHostC;
    fpHostA = (float*)malloc(nBytes);
    fpHostB = (float*)malloc(nBytes);
    fpHostC = (float*)malloc(nBytes);

    for (int i = 0; i<N*N; i++){
        fpHostA[i] = (float)(rand()%10);
        fpHostB[i] = (float)(rand()%10);
    }
    
    float * fpDeviceA, *fpDeviceB, *fpDeviceC;
    cudaMalloc(&fpDeviceA, nBytes);
    cudaMalloc(&fpDeviceB, nBytes);
    cudaMalloc(&fpDeviceC, nBytes);

    cudaMemcpy(fpDeviceA, fpHostA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDeviceB, fpHostB, nBytes, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS = N / THREADS;

    dim3 blockSize(THREADS,THREADS);
    dim3 gridSize(BLOCKS, BLOCKS);

    matrixMultiply<<<gridSize, blockSize>>>(fpDeviceA, fpDeviceB, fpDeviceC, N);
    
    cudaMemcpy(fpHostC, fpDeviceC, nBytes, cudaMemcpyDeviceToHost);

    verifyResult(fpHostA, fpHostB, fpHostC, N);

    printf("computed correctly\n");

    return 0;

}
