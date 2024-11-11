import torch 
from torch.utils.cpp_extension import load_inline

cuda_source = ""
with open('./inline_cuda.cu', 'r') as f:
    cuda_source = f.read()

# cuda_source = '''
# __global__ void square_matrix_kernel(const float* inputMatrix, const float* result, int width, int height){
#     int col = blockIdx.x * blockDim.x + threadIdx.x;
#     int row = blockIdx.y * blockDim.y + threadIdx.y;
    
#     if(row< height && col < width) {
#         int idx = row * width + col;
#         result[idx] = inputMatrix[idx] * inputMatrix[idx];
#     }
# }

# torch::Tensor square_matrix(torch::Tensor inputMatrix){
#     const auto width = inputMatrix.size(0);
#     const auto height = inputMatrix.size(1);

#     auto result = torch.empty_like(inputMatrix);
#     dim3 block(16,16);
#     dim3 grid((width+ block.x - 1) / block.x,
#             (height + block.y - 1) / block.y);
#     square_matrix_kernel<<<grid, block>>>(inputMatrix.data_ptr<float>(), result.data_ptr<float>(),width, height);
#     return result;
# }
# '''

cpp_source = "torch::Tensor square_matrix(torch::Tensor inputMatrix);"

square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=['-O2'],
    build_directory='./load_inline',
)
a = torch.tensor([[1.,2.,3.,],[4.,5.,6.,]], device='cuda')
print(square_matrix_extension.square_matrix(a))