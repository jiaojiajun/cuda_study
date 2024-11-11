import torch 

print(torch.cuda.is_available())


def cuda_time_pytorch(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up 
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def square_2(a):
    return a*a
def square_3(a):
    return a**2


a = torch.tensor([1.,2.,3.])
print(torch.square(a))
print(a*a)
print(a**2)


b = torch.randn(10000,10000).cuda()

cuda_time_pytorch(torch.square, b)
cuda_time_pytorch(square_2, b)
cuda_time_pytorch(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof: 
    torch.square(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print("=============")
print("Profiling a * a")
print("=============")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print("=============")
print("Profiling a ** 2")
print("=============")
with torch.autograd.profiler.profile(use_cuda=True) as prof: 
    square_3(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
