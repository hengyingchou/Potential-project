#import pycuda.autoinit
#import pycuda.driver as cuda 
from pycuda.compiler import SourceModule
mod = SourceModule("""
# define N 3
# define dx 0.01


	__global__ void RK4_addition
	(float*x1,float*x0,float*f1, float*f2 , float*f3, float*f4,float*par,float*x,float*y,float*z)
	{
		const int i = threadIdx.x+blockDim.x*blockIdx.x;
		const int j = int(par[0]);
		if(i < N){

			x1[i] = 0.0;
			x1[i] = x0[i]+(1.0/6.0)*(1.0*f1[i]+2.0*f2[i]+2.0*f3[i]+1.0*f4[i]);
			x0[i] = x1[i];
			if(i == 0) x[j] = x1[0];
			if(i == 1) y[j] = x1[1];
			if(i == 2) z[j] = x1[2];
		}
		par[0] = par[0]+1.0;
	}

	 __global__ void RK4_f1(float*A,float*x0,float*f1,float*x1){
                const int i = threadIdx.x+blockDim.x*blockIdx.x;
                __shared__ float temp[N];
                if(i < N ){
                        const int index = i*N;
			temp[i] = 0.0;
                        for(int j = 0 ; j < N ; j++){
                                temp[i] += A[index+j]*x0[j];
                        } 
                        f1[i] = temp[i]*dx;
                        x1[i] = x0[i] + temp[i]*dx*0.5;
                }
        }


	__global__ void RK4_f2(float*A,float*x0,float*f2,float*x1){
		const int i = threadIdx.x+blockDim.x*blockIdx.x;
		__shared__ float temp[N];
		if(i < N ){
			temp[i] = 0.0;
			const int index = i*N;
                	for(int j = 0 ; j < N ; j++){
                        	temp[i] += A[index+j]*x1[j];
                	} 
			f2[i] = temp[i]*dx;
			x1[i] = x0[i] + temp[i]*dx*0.5;
		}
	}

	__global__ void RK4_f3(float*A,float*x0,float*f3,float*x1){
                const int i = threadIdx.x+blockDim.x*blockIdx.x;
                __shared__ float temp[N];
                if(i < N ){
                        const int index = i*N;
			temp[i] = 0.0;
                        for(int j = 0 ; j < N ; j++){
                                temp[i] += A[index+j]*x1[j];
                        } 
                        f3[i] = temp[i]*dx;
                	x1[i] = x0[i] + temp[i]*dx;
		}
        }

	__global__ void RK4_f4(float*A,float*f4,float*x1){
                const int i = threadIdx.x+blockDim.x*blockIdx.x;
                __shared__ float temp[N];
                if(i < N ){
                        const int index = i*N;
			temp[i] = 0.0;
                        for(int j = 0 ; j < N ; j++){
                                temp[i] += A[index+j]*x1[j];
                        } 
                        f4[i] = temp[i]*dx;
                }
        }

	__global__ void New_A(float*A,float*x){
                A[5] = -x[0];
		A[6] = x[1];
        }


""")

