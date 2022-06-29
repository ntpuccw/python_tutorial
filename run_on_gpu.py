# Some benchmark examples to Test GPU and run scripts with and without GPU  
#%%
import torch
use_cuda = torch.cuda.is_available()

if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# %%
from numba import jit
import numpy as np
from timeit import default_timer as timer
N = 100_000_000 # N=1e8 time(CPU)/time(GPU)=200
# To run on CPU
def func(a):
    for i in range(N):
        a[i]+= 1
# To run on GPU
@jit   # Transforms the function into machine code 
def func2(a):
    for i in range(N):
        a[i]+= 1

if __name__=="__main__":
    n = N
    a = np.ones(n, dtype = np.float64)
    start = timer()
    # jit(func(a)) # useless
    func(a)
    print("without GPU:", timer() - start)
    start = timer()
    func2(a)
    # numba.cuda.profile_stop()
    print("with GPU:", timer() - start)

# %%
import numpy as np
from timeit import default_timer as timer
import numba  # We added these two lines for a 500x speedup

@numba.jit    # We added these two lines for a 500x speedup
def sum(x):
    total = 0
    for i in range(x.shape[0]):
        total += x[i]
    return total

n = 100000000
a = np.ones(n, dtype = np.float64)
start = timer()  
sum(a)
print("with GPU:", timer() - start)
# %%
def sum(x):
    total = 0
    for i in range(x.shape[0]):
        total += x[i]
    return total

n = 100000000
a = np.ones(n, dtype = np.float64)
start = timer()  
sum(a)
print("without GPU:", timer() - start)
# %% Approximate pi with and without GPU
import numba
import random
from timeit import default_timer as timer

@numba.jit() 
def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x ** 2 + y ** 2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples

start = timer()  
print(monte_carlo_pi(10_000_000))
print("Executing time:{:.2f}".format(timer()-start))
# %%
