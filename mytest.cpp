#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define TILE_WIDTH 16

#define CUDA_KERNEL_LOOP(i,n) \
for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
i < (n); \
i +=blockDim.x * gridDim.x)

template<typename T>
__global__ void MaxPoolForward(const T* bottom_data,  T* top_data,
const int nthreads, const int channels, const int height, const int width,
const int pooled_height, const int pooled_widht,const int kernel_h,const int kernel_w,
const int stride_h, const int stride_w, const int pad_h, const int pad_w, const int batch_size)
/*
bottom_data-input
top_data-output feature
nthreads- the number of top_data
*/
{
	CUDA_KERNEL_LOOP(index,nthreads){
		const int pw=index%pooled_widht;
		const int ph=(index/pooled_widht)%pooled_height;
		const int c=(index/pooled_widht/pooled_height)%channels;
		const int n=index/pooled_widht/pooled_height/channels;
		
		int hstart=ph*stride_h-pad_h;
		int wstart=pw*stride_w-pad_w;
		const int hend=min(hstart+kernel_h,height);
		const int wend=min(wstart+kernel_w,width);
		hstart=max(hstart,0);
		wstart=max(wstart,0);
		T maxval= -FLT_MAX;
		int maxidx = -1;
		const T* bottom_slice=bottom_data+(n*channels+c)*height*width;
		for(int h =hstart; h<hend; h++){
			for(int w=wstart;w<wend;w++){
				if(bottom_slice[h*width+w]>maxval){
					maxidx=h*width+w;
					maxval=bottom_slice[maxidx];
				}
			}
		}
		top_data[index]=maxval;	
	}
}

template<typename T>
__global__ void relu(const T* A, T* B,const int nthreads)
//A-input B-output
{
	CUDA_KERNEL_LOOP(index,nthreads){
		if(A[index]>0){
			B[index]=A[index];
		}else{
			B[index]=0;
		}
		
	}
}

template<typename T>
__global__ void add(const T* A,const T* B, T* C,const int nthreads)
//A-input B-input C-output
{
	CUDA_KERNEL_LOOP(index,nthreads){
		C[index]=A[index]+B[index];
	}
}


template<typename T>
__global__ void AvgPoolForward(const T* bottom_data,  T* top_data,
const int nthreads, const int channels, const int height, const int width,
const int pooled_height, const int pooled_widht,const int kernel_h,const int kernel_w,
const int stride_h, const int stride_w, const int pad_h, const int pad_w)
/*
bottom_data-input
top_data-output feature
nthreads- the number of top_data
*/
{
	CUDA_KERNEL_LOOP(index,nthreads){
		const int pw=index%pooled_widht;
		const int ph=(index/pooled_widht)%pooled_height;
		const int c=(index/pooled_widht/pooled_height)%channels;
		const int n=index/pooled_widht/pooled_height/channels;
		
		int hstart=ph*stride_h-pad_h;
		int wstart=pw*stride_w-pad_w;
		const int hend=min(hstart+kernel_h,height);
		const int wend=min(wstart+kernel_w,width);
		hstart=max(hstart,0);
		wstart=max(wstart,0);
		
		T tmp= 0;
		
		const T* bottom_slice=bottom_data+(n*channels+c)*height*width;
		for(int h =hstart; h<hend; h++){
			for(int w=wstart;w<wend;w++){
				tmp += bottom_slice[h*width+w];
			}
		}
		
		top_data[index]=tmp/((hend-hstart)*(wend-wstart));
		
		
	}
}

void matgen(float* a, int x, int y)
{
    int i, j;
    for (i = 0; i < x; i++)
    {
        for (j = 0; j < y; j++)
        {
            a[i * x + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
        }
    }
}

double Try_GPU()
{
    float *M, *N, *Pg;
    int x = 1024;	//1024×1024矩阵乘法
    int y = 1024;
    int z = 1024;
    M = (float*)malloc(sizeof(float)* x * y);
    N = (float*)malloc(sizeof(float)* y * z);
    Pg = (float*)malloc(sizeof(float)* x * z); //保存GPU计算结果

    srand(0);

    matgen(M, x, y);			//产生矩阵M
    matgen(N, y, z);			//产生矩阵N

    double timeStart, timeEnd;	//定义时间，求时间差用
    timeStart = clock();
    //MatrixMultiplication_CUDA(M, N, Pg, x, y, z, gamma);			//GPU上计算
    timeEnd = clock();

    free(M);
    free(N);
    free(Pg);
    return timeEnd - timeStart;
}

int main()
{
    printf("GPU use time %g\n", Try_GPU());
    system("pause");
    return 0;
}