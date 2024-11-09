#include<stdio.h>

void initData(float* fp, int elementsNum){
    if (fp==NULL){
        printf("init data error, null pointer\n");
    }
    memset(fp,0, elementsNum*sizeof(float));
    for (int i=0; i< elementsNum; i++){
        fp[i] = (float)(rand() & 0xff) / 10.0f;
    }

}
__global__ void plusInGpu(float* A, float*B, float*C, int elementsNum){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i< elementsNum){
        C[i] = A[i] + B[i];
    }
}


int main(){

    // 1. device check
    int deviceCnt = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCnt);
    if (error!=cudaSuccess || deviceCnt == 0){
        printf("can not find compatible device!\n");
        exit(-1);
    }else{
        printf("found device count: %d\n", deviceCnt);
    }

    int deviceId = 0;
    error = cudaSetDevice(deviceId);
    if( error!= cudaSuccess){
        printf("can not set device 0 for computation\n");
        exit(-1);
    }

    // 2. memory allocation
    float* fpHostA = NULL;
    float* fpHostB = NULL;
    float* fpHostC = NULL;

    int elementsNum = 2048;
    size_t nBytes = elementsNum * sizeof(float);
    fpHostA = (float*)malloc(nBytes);
    fpHostB = (float*)malloc(nBytes);
    fpHostC = (float*)malloc(nBytes);
    memset(fpHostA,0, nBytes);
    memset(fpHostB,0, nBytes);
    memset(fpHostC,0, nBytes);


    if(fpHostA == NULL || fpHostB == NULL || fpHostC == NULL){
        printf("failed to malloc host memory\n");
        exit(-1);
    }

    float* fpDeviceA =NULL;
    float* fpDeviceB =NULL;
    float* fpDeviceC =NULL;
    error = cudaMalloc(&fpDeviceA, nBytes);
    if (error!= cudaSuccess){
        printf("can not malloc size %d Bytes device memory for A\n", elementsNum);
    }
    error = cudaMalloc(&fpDeviceB, nBytes);
    if (error!= cudaSuccess){
        printf("can not malloc size %d Bytes device memory for B\n", elementsNum);
    }
    error = cudaMalloc(&fpDeviceC, nBytes);
    if (error!= cudaSuccess){
        printf("can not malloc size %d Bytes device memory for C\n", elementsNum);
    }
    cudaMemset(fpDeviceA,0,nBytes);
    cudaMemset(fpDeviceB,0,nBytes);
    cudaMemset(fpDeviceC,0,nBytes);


    // 3. init data in host 
    srand(123);
    initData(fpHostA, elementsNum);
    initData(fpHostB, elementsNum);
    for (int i = 0; i < 10; i++){
        printf("%f\t",fpHostA[i]);
    }
    

    // 4. copy data to device
    error = cudaMemcpy(fpDeviceA, fpHostA, nBytes, cudaMemcpyHostToDevice);
    if (error!=cudaSuccess){
        printf("failed to copy data from host to device");
        exit(-1);
    }
    error = cudaMemcpy(fpDeviceB, fpHostB, nBytes, cudaMemcpyHostToDevice);
    
    
    // 5. call kernel function
    dim3 block(256);
    dim3 grid((elementsNum + 256-1) /256);
    plusInGpu<<<grid, block>>>(fpDeviceA, fpDeviceB, fpDeviceC, elementsNum);
    cudaDeviceSynchronize();

    // 6. copy data to host memory 
    cudaMemcpy(fpHostC, fpDeviceC, nBytes, cudaMemcpyDeviceToHost);

    for(int i=0;i< 10; i++){
        printf("c[%d] is %f\t",i, fpHostC[i]);
    }

    // 7. free memory
    free(fpHostA);
    free(fpHostB);
    free(fpHostC);

    cudaFree(fpDeviceA);
    cudaFree(fpDeviceB);
    cudaFree(fpDeviceC);
    return 0;

}