#include<stdio.h>
#include"../5_error_check/common.cuh"

int main(){
    cudaEvent_t start, stop;
    errorCheck(cudaEventCreate(&start),__FILE__, __LINE__);
    errorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
    errorCheck(cudaEventRecord(start),__FILE__, __LINE__);
    cudaEventQuery(start);

    // wast time code 

    errorCheck(cudaEventRecord(stop),__FILE__,__LINE__);
    errorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);

    float time_elapsed;
    errorCheck(cudaEventElapsedTime(&time_elapsed, start, stop),__FILE__, __LINE__);
    printf("time elapsed %g ms.\n", time_elapsed);
    return 0;

}