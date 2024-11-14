#include<stdio.h>
#include<cassert>

const int NARRAY = 1<<20;
const int NKERNEL = 10;

__constant__ float kernel[NKERNEL];

__global__ void convolution(float* array, float*res){
    int r = NKERNEL / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid - r;
    float tmp = 0;
    for (int i = 0; i < NKERNEL; i++){
        if ((start + i) >=0 && (start + i) < NARRAY){
            tmp += array[start+i] * kernel[i];
        }
        
    }
    res[tid] = tmp;
}

void verifyResult(float* array, float* kernel, float* res){
    int r = NKERNEL / 2;
    float eps = 1e-2;
    for (int i = 0; i < NARRAY; i++){
        int start = i - r;
        float tmp = 0;
        for (int j = 0; j < NKERNEL; j++){
            if ((start + j) >= 0 && (start + j) < NARRAY){
                tmp += array[start + j] * kernel[j];
            }
        }
        if (fabs(tmp - res[i]) >= eps){
            printf("tmp is %f, while res is %f, the fab is %f\n",tmp, res[i], eps);
        }
        // assert( fabs(tmp - res[i]) < eps );
    }
}

int main(){
    const size_t nArrayBytes = NARRAY * sizeof(float);
    const size_t nKernelBytes = NKERNEL *sizeof(float);

    float* fpHostArray, *fpDeviceArray, *fpHostKernel;
    fpHostArray = (float*)malloc(nArrayBytes);
    fpHostKernel = (float*)malloc(nKernelBytes);

    cudaMalloc(&fpDeviceArray, nArrayBytes);


    float* fpHostRes, *fpDeviceRes;
    fpHostRes = (float*) malloc(nArrayBytes);
    cudaMalloc(&fpDeviceRes, nArrayBytes);

    for(int i = 0; i < NARRAY; i++){
        fpHostArray[i] = (float)((float)(rand()%100)/10.0f);
    }
    for(int i = 0; i < NKERNEL; i++){
        fpHostKernel[i] = (float)(rand()%10);
    }

    cudaMemcpy(fpDeviceArray, fpHostArray, nArrayBytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(kernel, fpHostKernel, nKernelBytes);

    int blockSize = 256 ;
    int gridSize = (NARRAY + blockSize - 1) /blockSize;

    convolution<<<gridSize, blockSize>>>(fpDeviceArray, fpDeviceRes);
    cudaMemcpy(fpHostRes, fpDeviceRes, nArrayBytes, cudaMemcpyDeviceToHost);

    verifyResult(fpHostArray, fpHostKernel, fpHostRes);

    printf("compute successfully\n");

    return 0;

}