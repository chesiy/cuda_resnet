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

template<typename T>
__global__ void MaxPoolForward(const T* bottom_data, T* top_data,
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
__global__ void AvgPoolForward(const T* bottom_data, const T* top_data,
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
        int maxidx = -1;

        const T* bottom_slice=bottom_data+(n*channels+c)*height*width;
        for(int h =hstart; h<hend; h++){
            for(int w=wstart;w<wend;w++){
                tmp += bottom_slice[h*width+w];
                /*if(bottom_slice[h*width+w]>maxval){
                    maxidx=h*width+w;
                    maxval=bottom_slice[maxidx];
                }*/
            }
        }

        top_data[index]=tmp/((hend-hstart)*(wend-wstart));

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
        int cur_batch = index / batch_size / out_numrow / out_numcol;
        int cur_c = (index / out_numrow / out_numcol) % batch_size;
        int cur_row = (index / out_numcol) % out_numrow;
        int cur_col = index % out_numcol;

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
            float* A_slice = A_b + cur_batch*in_channels*out_numrow*out_numcol + cur_inp_c*out_numrow*out_numcol;
            float* kernel_slice = kernel + cur_c*in_channels*kernel_numrow*kernel_numcol + cur_inp_c*kernel_numrow*kernel_numcol;
            for(int i=0; i<ker_end_row-ker_start_row; i++){
                for(int j=0; j<ker_end_col-ker_start_col; j++){
                    tmp += A_slice[(start_row+i)*in_numcol + (start_col+j)] * kernel_slice[(ker_start_row+i)*kernel_numcol + (ker_start_col+j)];
                }
            }
        }
        tmp += bias[cur_batch*out_channels*out_numrow*out_numcol + cur_c*out_numrow*out_numcol + cur_row*out_numcol + cur_col]; // bias[cur_batch, cur_c, cur_row, cur_col]
        C_b[cur_batch*out_channels*out_numrow*out_numcol + cur_c*out_numrow*out_numcol + cur_row*out_numcol + cur_col] = tmp; // C_b[cur_batch, cur_c, cur_row, cur_col]
    }
}


//__global__ void batch_trivial_conv2d_square_kernel(float* A, float*kernel, float*C, float* bias,
//                                                   int batch_size, int in_numrow, int in_numcol, int in_channels, int kernel_size, int out_channels,
//                                                   int padding=0, int stride=1)
//// use square kernel and padding, and use same stride in col and row
//// may be a little easier to implement ?
//{
//    return ConvolutionForward(A, kernel, C, bias,
//                                batch_size, in_numrow, in_numcol, in_channels, kernel_size, kernel_size, out_channels,
//                                padding, padding, stride, stride);
//}


__global__ void trivial_conv2d(float* A, float*kernel, float*C, float* bias, int nthreads,
                               int in_numrow, int in_numcol, int in_channels, int kernel_numrow, int kernel_numcol, int out_channels,
                               int row_padding=0, int col_padding=0, int stride_row=1, int stride_col=1)
{
    int out_numrow = (in_numrow + row_padding*2 - kernel_numrow) / stride_row + 1;
    int out_numcol = (in_numcol + col_padding*2 - kernel_numcol) / stride_col + 1;
    // A: in_channels x in_numrow x in_numcol
    // kernel: out_channels x in_channels x kernel_numrow x kernel_numcol
    // B: out_channels x out_numrow x out_numcol
    // bias: out_channels x out_numrow x out_numcol

    CUDA_KERNEL_LOOP(index, nthreads){
        int cur_c = index / (out_numrow*out_numcol);
        int cur_row = (index / out_numcol) % out_numrow;
        int cur_col = index % out_numcol;

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
            float* A_slice = A + cur_inp_c*out_numrow*out_numcol; // A[cur_inp_c, :]
            float* kernel_slice = kernel + cur_c*in_channels*kernel_numrow*kernel_numcol +
                                  cur_inp_c*kernel_numrow*kernel_numcol; // kernel[cur_c, cur_inp_c, :]
            for(int i=0; i<ker_end_row-ker_start_row; i++){
                for(int j=0; j<ker_end_col-ker_start_col; j++){
                    tmp += A_slice[(start_row+i)*in_numcol + (start_col+j)] * kernel_slice[(ker_start_row+i)*kernel_numcol + (ker_start_col+j)];
                }
            }
        }
        tmp += bias[cur_c*out_numrow*out_numcol + cur_row*out_numcol + cur_col];
        C[cur_c*out_numrow*out_numcol + cur_row*out_numcol + cur_col] = tmp;
    }
}