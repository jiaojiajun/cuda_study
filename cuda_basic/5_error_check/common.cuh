#pragma once

#include<stdio.h>

cudaError_t errorCheck(cudaError_t error, const char* fileName, const int lineNumber){
    if(error != cudaSuccess){
        printf("cuda error:\r\ncode=%d, name=%s, description=%s\r\n file=%s, line=%d\r\n",
        error, cudaGetErrorName(error), cudaGetErrorString(error),fileName, lineNumber);
        return error;
    }
    return cudaSuccess;
}