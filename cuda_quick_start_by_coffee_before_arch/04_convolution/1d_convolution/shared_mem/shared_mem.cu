#include<stdio.h>
#include<cassert>

const int nArray = 1<<20;
const int nKernel = 7;

__constant__ float kernel[nKernel];

__global__ void convolution(float* array, float* res){
    int r = nKernel / 2;
    int d = r * 2;
    const int n_p = blockDim.x + d;
    extern __shared__ float sharedMem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = threadIdx.x + blockDim.x;
    sharedMem[threadIdx.x] = array[tid];
    if(offset < n_p){
        sharedMem[offset] = array[blockIdx.x * blockDim.x + offset];
    } 
    __syncthreads();
    float tmp = 0.0f;
    for (int i = 0; i < nKernel; i++){
        tmp += sharedMem[threadIdx.x + i] * kernel[i]; 
    }
    res[tid] = tmp;
}

void verifyResult(float* array, float* kernel, float* res){
    float tmp = 0;
    float eps = 1e-3;
    for (int i = 0; i < nArray; i++){
        tmp =0;
        for (int j = 0; j < nKernel; j++){
            tmp += array[i+j] * kernel[j];
        }
        assert(fabs(tmp-res[i]) < eps);
        if (fabs(tmp-res[i]) >= eps){
            printf("tmp is %f, while res is %f\n", tmp, res[i]);
        }
    }
}

int main(){
    float * fpHostArray, *fpHostKernel, *fpHostRes;
    const int r = nKernel / 2;
    const int n_p = nArray + 2 * r;
    const size_t nArrayBytes = n_p * sizeof(float);
    fpHostArray = (float*)malloc(nArrayBytes);
    fpHostKernel = (float*)malloc(nKernel * sizeof(float));
    fpHostRes = (float*)malloc(nArrayBytes);


    float *fpDeviceArray, *fpDeviceRes;
    cudaMalloc(&fpDeviceArray, nArrayBytes);
    cudaMalloc(&fpDeviceRes, nArrayBytes);
    
    for (int i = 0; i < n_p; i++){ // 10.5f 没有特殊含义，仅仅是随机初始化的一个数值，下面一样。
        if(i < r || i >= r + nArray){
            fpHostArray[i] = 0.0f;
        }else{
            fpHostArray[i] = (rand()%100)/ 10.5f;
        }
    }
    for (int i = 0; i < nKernel; i++){
        fpHostKernel[i] = (rand()%100)/10.5f;
    }
    cudaMemcpyToSymbol(kernel, fpHostKernel,nKernel*sizeof(float));

    cudaMemcpy(fpDeviceArray, fpHostArray, nArrayBytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (nArray + blockSize - 1) / blockSize;
    const int SHEMEMSIZE = (blockSize + r * 2) * sizeof(float);
    convolution<<<gridSize, blockSize, SHEMEMSIZE>>> (fpDeviceArray, fpDeviceRes);
    cudaMemcpy(fpHostRes, fpDeviceRes, nArrayBytes, cudaMemcpyDeviceToHost);

    verifyResult(fpHostArray, fpHostKernel, fpHostRes);

    printf("compute successfully\n");

    return 0;
}