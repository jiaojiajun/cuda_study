#include<stdio.h>
#include<cassert>

const int N = 1<< 10;
const int SHMEM_SIZE = 1<<10;


__global__ void matrix_multiply(float* A,float* B,float*C){
    int row =blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedMemA[SHMEM_SIZE];
    __shared__ float sharedMemB[SHMEM_SIZE];

    float tmp =0;
    for (int i=0; i<N; i+= blockDim.x){
        sharedMemA[threadIdx.y * blockDim.x + threadIdx.x] = A[row *N + i + threadIdx.x];
        sharedMemB[threadIdx.y * blockDim.x + threadIdx.x] = B[i*N + threadIdx.y*N + col];
        __syncthreads();
        for(int j=0; j<blockDim.x; j++){
            tmp += sharedMemA[threadIdx.y * blockDim.x + j] * sharedMemB[j* blockDim.x + threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row*N+col] = tmp;
}

void verifyResult(float* A,float* B,float*C){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float tmp = 0.0f;
            for (int k = 0; k < N; k++){
                tmp+= A[i*N +k] * B[k*N + j];
            }
            // printf("c[i][j] is %f, while tmp is %f\n", C[i*N+j],tmp);
            assert(tmp == C[i*N+j]);
        }
    }
}


int main(){
    const size_t nBytes = N * N *sizeof(float);
    float * fpHostA, *fpHostB, *fpHostC;
    float *fpDeviceA, *fpDeviceB, *fpDeviceC; 
    fpHostA = (float*)malloc(nBytes);
    fpHostB = (float*)malloc(nBytes);
    fpHostC = (float*)malloc(nBytes);

    cudaMalloc(&fpDeviceA, nBytes);
    cudaMalloc(&fpDeviceB, nBytes);
    cudaMalloc(&fpDeviceC, nBytes);

    for (int i=0; i<N*N; i++){
        fpHostA[i] = (float)(rand()%10);
        fpHostB[i] = (float)(rand()%10);
    }
    cudaMemcpy(fpDeviceA, fpHostA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(fpDeviceB, fpHostB, nBytes, cudaMemcpyHostToDevice);

    const int THREADS = 32;
    const int BLOCKS = N/32;
    dim3 block_size(THREADS, THREADS);
    dim3 grid_size(BLOCKS, BLOCKS);
    matrix_multiply<<<grid_size, block_size>>>(fpDeviceA, fpDeviceB,fpDeviceC);
    cudaMemcpy(fpHostC, fpDeviceC, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(fpDeviceA);
    cudaFree(fpDeviceB);
    cudaFree(fpDeviceC);


    verifyResult(fpHostA, fpHostB, fpHostC);
    printf("compute successful\n");
    return 0;
}