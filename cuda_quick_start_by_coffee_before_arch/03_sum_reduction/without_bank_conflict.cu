#include<stdio.h>
#include<cassert>

const int N = 1 << 16;
const size_t nBytes = N * sizeof(int);
const int SIZE = 256;

void initVecotor(int* p){
    for (int i = 0; i < N; i++){
        p[i] = 1;
    }
}

void verifyResult(int *p){
    printf("the res is %d\n", p[0]);
}

__global__ void sumReduce(int *p, int * res){
    const int MEM_SIZE = SIZE * 4; 
    __shared__ int sharedMem[MEM_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    sharedMem[threadIdx.x] = p[tid];
    __syncthreads();
    for(int s = blockDim.x /2; s > 0; s>>=1){
        if(threadIdx.x < s){
            sharedMem[threadIdx.x ] += sharedMem[threadIdx.x+s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        res[blockIdx.x] = sharedMem[0];
    }
}

int main(){
    int *ipHost,*ipHostRes, *ipDevice, *ipDeviceRes;
    ipHost = (int*) malloc(nBytes);
    ipHostRes = (int*) malloc(nBytes);

    cudaMalloc(&ipDevice, nBytes);
    cudaMalloc(&ipDeviceRes, nBytes);
    initVecotor(ipHost);
    cudaMemcpy(ipDevice, ipHost, nBytes, cudaMemcpyHostToDevice);

    int THREADS = SIZE;
    int BLOCKS = N / SIZE;

    sumReduce<<<BLOCKS, THREADS>>>(ipDevice,ipDeviceRes); 
    sumReduce<<<1,BLOCKS>>>(ipDeviceRes, ipDeviceRes);
    
    cudaMemcpy(ipHostRes, ipDeviceRes, sizeof(int), cudaMemcpyDeviceToHost);

    verifyResult(ipHostRes);

    return 0;
}