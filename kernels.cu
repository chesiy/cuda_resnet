#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define CUDA_KERNEL_LOOP(i,n) \
for(int i = blockIdx.x * blockDim.x + threadIdx.x; \
i < (n); \
i +=blockDim.x * gridDim.x)

#define FLT_MAX 999999999999

int Min(int a,int b){
    return a<b ? a:b;
}

int Max(int a,int b){
    return a>b ? a:b;
}

__global__ void MaxPoolForward(const float* bottom_data, float* top_data,
                               const int nthreads, const int channels, const int height, const int width,
                               const int pooled_height, const int pooled_width,const int kernel_h,const int kernel_w,
                               const int stride_h, const int stride_w, const int pad_h, const int pad_w)
/*
bottom_data-input
top_data-output feature
nthreads- the number of top_data
*/
{
    CUDA_KERNEL_LOOP(index,nthreads){
        const int pw=index%pooled_width;
        const int ph=(index/pooled_width)%pooled_height;
        const int c=(index/pooled_width/pooled_height)%channels;
        const int n=index/pooled_width/pooled_height/channels;

        int hstart=ph*stride_h-pad_h;
        int wstart=pw*stride_w-pad_w;
        const int hend=min(hstart+kernel_h,height);
        const int wend=min(wstart+kernel_w,width);
        hstart=max(hstart,0);
        wstart=max(wstart,0);
        float maxval= -FLT_MAX;
        int maxidx = -1;
        const float* bottom_slice=bottom_data+(n*channels+c)*height*width;
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

__global__ void relu(const float* A, float* B,const int nthreads)
//A-input B-output
{
    CUDA_KERNEL_LOOP(index,nthreads){
        if(A[index]>0){
            B[index]=A[index];
        }else{
            B[index]=0;
        }
//        printf("%f ",B[index]);
    }
}


__global__ void add_relu(const float* A,const float* B, float* C,const int nthreads)
//A-input B-input C-output
{
    CUDA_KERNEL_LOOP(index,nthreads){
        C[index]=A[index]+B[index];
        if(C[index]>0){
            C[index]=C[index];
        }else{
            C[index]=0;
        }
    }
}

__global__ void add(const float* A,const float* B, float* C,const int nthreads)
//A-input B-input C-output
{
    CUDA_KERNEL_LOOP(index,nthreads){
        C[index]=A[index]+B[index];
    }
}

__global__ void AvgPoolForward(const float* bottom_data, float* top_data,
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
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);

        const int pool_size = (hend - hstart) * (wend - wstart);

        hstart=max(hstart,0);
        wstart=max(wstart,0);
        hend = min(hend, height);
        wend = min(wend, width);

        float tmp= 0;

        const float* bottom_slice=bottom_data+(n*channels+c)*height*width;
        for(int h =hstart; h<hend; h++){
            for(int w=wstart;w<wend;w++){
                tmp += bottom_slice[h*width+w];
            }
        }

        top_data[index]=tmp/pool_size ;

    }
}

// write two most trivial algorithms
// in further use, out_numrow and out_numcol should be calculated first to save time
__global__ void ConvolutionForward(float* A_b, float*C_b, float*kernel, float* bias, int nthreads,
                                   int batch_size, int in_numrow, int in_numcol, int in_channels,
                                   int out_numrow, int out_numcol, int out_channels,
                                   int kernel_numrow, int kernel_numcol,
                                   int stride_row=1, int stride_col=1,int row_padding=0, int col_padding=0)
{
//    int out_numrow = (in_numrow + row_padding*2 - kernel_numrow) / stride_row + 1;
//    int out_numcol = (in_numcol + col_padding*2 - kernel_numcol) / stride_col + 1;
    // A_b: batch_size x in_channels x in_numrow x in_numcol
    // kernel: out_channels x in_channels x kernel_numrow x kernel_numcol
    // B_b: batch_size x out_channels x out_numrow x out_numcol
    // bias: batch_size x out_channels x out_numrow x out_numcol

    CUDA_KERNEL_LOOP(index, nthreads){
//        printf("kernel... %f %f", A_b[10], A_b[20]);
        int cur_batch = index / out_channels / out_numrow / out_numcol;
        int cur_c = (index / out_numrow / out_numcol) % out_channels;
        int cur_row = (index / out_numcol) % out_numrow;
        int cur_col = index % out_numcol;

        // printf("%d ", cur_batch);

        int start_row = cur_row * stride_row - row_padding; // start row in input
        int end_row = start_row + kernel_numrow;

        int start_col = cur_col * stride_col - col_padding; // start column in input
        int end_col = start_col + kernel_numcol; // end_col is not included

        // deal with padding, only use zero-padding
        int ker_start_row = 0, ker_end_row = kernel_numrow;
        int ker_start_col = 0, ker_end_col = kernel_numcol;
        if(start_row < 0){
            ker_start_row = - start_row;
            start_row = 0;
        }
        if(start_col < 0){
            ker_start_col = - start_col;
            start_col = 0;
        }

        if(end_row > in_numrow){
            ker_end_row = ker_end_row - end_row + in_numrow;
            end_row = in_numrow;
        }
        if(end_col > in_numcol){
            ker_end_col = ker_end_col - end_col + in_numcol;
            end_col = in_numcol;
        }

        float tmp = 0;
        for(int cur_inp_c=0; cur_inp_c<in_channels; cur_inp_c++){ // for each input channel
            float* A_slice = A_b + cur_batch*in_channels*in_numrow*in_numcol + cur_inp_c*in_numrow*in_numcol;
            float* kernel_slice = kernel + cur_c*in_channels*kernel_numrow*kernel_numcol + cur_inp_c*kernel_numrow*kernel_numcol;
            for(int i=0; i<ker_end_row-ker_start_row; i++){
                for(int j=0; j<ker_end_col-ker_start_col; j++){
                    tmp += A_slice[(start_row+i)*in_numcol + (start_col+j)] * kernel_slice[(ker_start_row+i)*kernel_numcol + (ker_start_col+j)];
                }
            }
        }
        tmp += bias[cur_c];
//        printf("tmp:%f , sitation: %d \n",tmp,cur_batch*out_channels*out_numrow*out_numcol + cur_c*out_numrow*out_numcol + cur_row*out_numcol + cur_col);

        C_b[cur_batch*out_channels*out_numrow*out_numcol + cur_c*out_numrow*out_numcol + cur_row*out_numcol + cur_col] = tmp; // C_b[cur_batch, cur_c, cur_row, cur_col]
    }
}


__global__ void serial_matmul(float* A, float*B, float*C,int dim_1, int dim_2, int dim_3) {
    // A: dim1 x dim2, B: dim2 x dim3, C: dim1 x dim3
    for (int i = 0; i < dim_1; i++) {
        for (int j = 0; j < dim_3; j++) {
            float tmp = 0;
            for (int k = 0; k < dim_2; k++) {
                tmp += A[i * dim_2 + k] * B[k * dim_3 + j];
            }
            C[i * dim_3 + j] = tmp;
        }
    }
}

__global__ void simple_matmul(const float* A, float* B, const float* Weight, const float* Bias,
                              const int nthreads, const int dim1, const int dim2, const int dim3){
    // A: dim1 x dim2, Weight: dim3 x dim2, B: dim1 x dim3 (Weight has been transposed) Bias: dim3
    // A x Weight + Bias = B
    CUDA_KERNEL_LOOP(index, nthreads){
        int cur_row = index / dim3;
        int cur_col = index % dim3;
//        printf("cur: %d %d %d\n", cur_row, cur_col, index);
        float tmp = 0;
        for(int i=0;i < dim2; i++){
            tmp +=  A[cur_row*dim2+i] * Weight[cur_col*dim2+i];
//            printf("curB:%d row:%d col:%d A: %f W: %f\n",index,cur_row,cur_col, A[cur_row*dim2+i],Weight[cur_col*dim2+i]);
        }
        B[cur_row * dim3 + cur_col] = tmp + Bias[cur_col];
//        printf("B: %f %f %f\n",tmp, Bias[cur_col], B[cur_row*dim3+cur_col]);
    }
}

