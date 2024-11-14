#include<stdio.h>
#include<cassert>

const int nArray = 1 << 20;
const int nKernel = 7;
const int r = nKernel / 2;
const int d = r * 2;
const int nPadding = nArray + d;
const size_t nPaddingBytes = nPadding * sizeof(float);
const size_t nKernelBytes = nKernel * sizeof(float);
const size_t nArrayBytes = nArray * sizeof(float);
__constant__ float kernel[nKernel];


__global__ void convolution(float* array, float* result){
    int tid = blockIdx.x  * blockDim.x + threadIdx.x;
    extern __shared__ float sharedMem[];
    sharedMem[threadIdx.x] = array[tid];
    __syncthreads();

    float tmp = 0.f;
    for (int i = 0; i< nKernel; i++){
        if (threadIdx.x + i >= blockDim.x){
            tmp += array[tid + i] * kernel[i];
        }else {
            tmp += sharedMem[threadIdx.x + i] * kernel[i];
        }
    }
    result[tid] = tmp;
}

void verifyResult(float* array, float* kernel, float* result){
    float eps = 1e-3;
    for(int i = 0; i < nArray; i++){
        float tmp = 0.f;
        for (int j = 0; j < nKernel; j++){
            tmp += array[i+j] * kernel[j];
        }
        assert(fabs(tmp - result[i]) < eps);
        // if (fabs(tmp - result[i]) >= eps){
        //     printf("tmp is %f, while result is %f and diff is %f, eps is %f\n", tmp, result[i], fabs(tmp - result[i]), eps);
        // }
    }
}

int main(){

    float *fpHostArrary, *fpHostKernel, *fpHostRes;
    fpHostArrary = (float*) malloc(nPaddingBytes);
    fpHostKernel = (float *) malloc(nKernelBytes);
    fpHostRes = (float*) malloc(nArrayBytes);

    float *fpDeviceArray, *fpDeviceRes;
    cudaMalloc(&fpDeviceArray, nPaddingBytes);
    cudaMalloc(&fpDeviceRes, nArrayBytes);

    for (int i = 0; i < nPadding; i++){
        if (i < r || i >= r + nArray){
            fpHostArrary[i] = 0;
        }else{
            fpHostArrary[i] = rand()%100 /10.5f;
        }
    }
    for ( int i = 0; i < nKernel; i++){
        fpHostKernel[i] = rand()%100/ 10.5f;
    }
    cudaMemcpy(fpDeviceArray, fpHostArrary, nPaddingBytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel, fpHostKernel, nKernelBytes);

    int blockSize = 256;
    int gridSize = (nPadding + blockSize - 1 ) / blockSize;
    const size_t SHMEM = blockSize * sizeof(float);
    convolution<<<gridSize, blockSize, SHMEM>>>(fpDeviceArray, fpDeviceRes);
    
    cudaMemcpy(fpHostRes, fpDeviceRes, nArrayBytes, cudaMemcpyDeviceToHost);

    verifyResult(fpHostArrary, fpHostKernel, fpHostRes);
    printf("compute successfully\n");
    return 0;
}