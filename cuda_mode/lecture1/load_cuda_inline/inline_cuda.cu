__global__ void square_matrix_kernel(const float* inputMatrix, float* result, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row< height && col < width) {
        int idx = row * width + col;
        result[idx] = inputMatrix[idx] * inputMatrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor inputMatrix){
    const auto width = inputMatrix.size(0);
    const auto height = inputMatrix.size(1);

    auto result = torch::empty_like(inputMatrix);
    dim3 block(16,16);
    dim3 grid((width+ block.x - 1) / block.x,
            (height + block.y - 1) / block.y);
    square_matrix_kernel<<<grid, block>>>(inputMatrix.data_ptr<float>(), result.data_ptr<float>(),width, height);
    return result;
}
