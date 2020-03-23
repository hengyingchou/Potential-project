import pycuda.driver as cuda 
import pycuda.autoinit
import cuda_code as code
from  cuda_code import  mod
import numpy as np
import math 
from timeit import default_timer as timer

#init data 
N = 3
steps = 5000
A = np.random.randn(N*N).astype(np.float32)
x0 = np.random.randn(N).astype(np.float32)
x_1 = np.random.randn(steps).astype(np.float32)
y_1 = np.random.randn(steps).astype(np.float32)
z_1 = np.random.randn(steps).astype(np.float32)
par = np.random.randn(1).astype(np.float32)
par[0] = 0.0
print('initial data success')

sigma = 10.0
beta = 2.666667
zo = 28
A[0] = -sigma
A[1] = sigma
A[2] = 0.0
A[3] = zo
A[4] = -1.0
A[5] = 1.0
A[6] = 1.0
A[7] = 0.0
A[8] = -beta

x0[0] = 1.0
x0[1] = 1.0
x0[2] = 1.0 

#allocate data in gpu.
A_gpu = cuda.mem_alloc(A.nbytes)
x0_gpu = cuda.mem_alloc(x0.nbytes)
x1 = cuda.mem_alloc(x0.nbytes)
f1 = cuda.mem_alloc(x0.nbytes)
f2 = cuda.mem_alloc(x0.nbytes)
f3 = cuda.mem_alloc(x0.nbytes)
f4 = cuda.mem_alloc(x0.nbytes)
gpu_x = cuda.mem_alloc(x_1.nbytes)
gpu_y = cuda.mem_alloc(y_1.nbytes)
gpu_z = cuda.mem_alloc(z_1.nbytes)
parr = cuda.mem_alloc(par.nbytes)


cuda.memcpy_htod(A_gpu,A)
cuda.memcpy_htod(x0_gpu,x0)
cuda.memcpy_htod(parr,par)
print('passing data to host successfully')

# Do parallel computation here. 

RK4_1 = mod.get_function("RK4_f1")
RK4_2 = mod.get_function("RK4_f2")
RK4_3 = mod.get_function("RK4_f3")
RK4_4 = mod.get_function("RK4_f4")
RK4_add = mod.get_function("RK4_addition")
RK4_A_up = mod.get_function("New_A")

start = timer()
for i in range(steps):
	RK4_A_up(A_gpu,x0_gpu,block = (1,1,1),grid = (1,1))
	RK4_1(A_gpu,x0_gpu,f1,x1,block = (N,1,1),grid = (1,1))

	RK4_A_up(A_gpu,x1,block = (1,1,1),grid = (1,1))
	RK4_2(A_gpu,x0_gpu,f2,x1,block = (N,1,1),grid = (1,1))

	RK4_A_up(A_gpu,x1,block = (1,1,1),grid = (1,1))
	RK4_3(A_gpu,x0_gpu,f3,x1,block = (N,1,1),grid = (1,1))

	RK4_A_up(A_gpu,x1,block = (1,1,1),grid = (1,1))
	RK4_4(A_gpu,f4,x1,block = (N,1,1),grid = (1,1))

	RK4_add(x1,x0_gpu,f1, f2 , f3, f4,parr,gpu_x,gpu_y,gpu_z,block = (N,1,1),grid = (1,1))

end = timer()
print("It takes",end-start,'seconds')
cuda.memcpy_dtoh(x_1,gpu_x)
cuda.memcpy_dtoh(y_1,gpu_y)
cuda.memcpy_dtoh(z_1,gpu_z)
np.savetxt('x.txt', x_1, delimiter=',')
np.savetxt('y.txt', y_1, delimiter=',')
np.savetxt('z.txt', z_1, delimiter=',')

A = []
x0 = []
result = []
par =[]

A_gpu.free() 
x0_gpu.free()
x1.free()
f1.free()
f2.free()
f3.free()
f4.free()
parr.free()
print("All memory freeing access")


