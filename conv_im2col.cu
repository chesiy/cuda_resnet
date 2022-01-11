#include <stdio.h>
#include <iostream>
#include <math.h>

namespace im2col{
    const int mm_tilewidth = 8; // tile width when do matrix multiply
    // when input is CHW format
    __global__ void im2col_CHW(float* A, float* out, int kernel_size, int in_channel, int in_numrow, int in_numcol,
                               int out_rownum, int out_colnum, int stride, int padding, int batch_size){
        // blocks: batch_size x (out_rownum x in_channel) x K, threads: out_colnum, K
        // out: batch_size x (in_channel * K * K) x (out_rownum * out_colnum)
        // use the strange format in order for Memory Coalescing

        int cur_out_row = blockIdx.y / in_channel, cur_out_col = threadIdx.x, cur_in_channel = blockIdx.y % in_channel;
        int cur_kernel_row = blockIdx.z, cur_kernel_col = threadIdx.y;

        int cur_inp_row = -padding + stride * cur_out_row + cur_kernel_row,
                cur_inp_col = -padding + stride * cur_out_col + cur_kernel_col; // place in input HxW image feature

        int cur_mat_col = cur_out_row * out_colnum + cur_out_col; // place in the row in out mat
        // place in the column in out mat
        int cur_mat_row = cur_in_channel * kernel_size * kernel_size + cur_kernel_row * kernel_size + cur_kernel_col;

        // out[batch_size, cur_mat_row, cur_mat_col] = A[batch_size, threadIdx.x, cur_inp_row+threadIdx.y, cur_inp_col+threadIdx.z]
        float tmp = 0;
        if(cur_inp_row >= 0 && cur_inp_row < in_numrow && cur_inp_col >= 0  && cur_inp_col < in_numcol){
            tmp = A[blockIdx.x*in_channel*in_numrow*in_numcol + cur_in_channel*in_numrow*in_numcol + cur_inp_row*in_numcol + cur_inp_col];
        }
        int row_len = in_channel * kernel_size * kernel_size, col_len = out_rownum * out_colnum;
        int idx = blockIdx.x * row_len * col_len + cur_mat_row * col_len + cur_mat_col;
        out[idx] = tmp;
    }

    __global__ void im2col_CHW_2(float* A, float* out, int kernel_size, int in_channel, int in_numrow, int in_numcol,
                                int out_rownum, int out_colnum, int stride, int padding, int batch_size){
        // blocks: batch_size x out_rownum x in_channel, threads: (out_colnum, K, K)
        // out: batch_size x (in_channel * K * K) x (out_rownum * out_colnum)
        // use the strange format in order for Memory Coalescing

        int cur_out_row = blockIdx.y, cur_out_col = threadIdx.x, cur_in_channel = blockIdx.z;
        // int cur_kernel_row = blockIdx.z / kernel_size, cur_kernel_col = blockIdx.z % kernel_size;
        int cur_kernel_row = threadIdx.y, cur_kernel_col = threadIdx.z;

        int cur_inp_row = -padding + stride * cur_out_row + cur_kernel_row,
                cur_inp_col = -padding + stride * cur_out_col + cur_kernel_col; // place in input HxW image feature

        int cur_mat_col = cur_out_row * out_colnum + cur_out_col; // place in the row in out mat
        // place in the column in out mat
        int cur_mat_row = cur_in_channel * kernel_size * kernel_size + cur_kernel_row * kernel_size + cur_kernel_col;

        // out[batch_size, cur_mat_row, cur_mat_col] = A[batch_size, threadIdx.x, cur_inp_row+threadIdx.y, cur_inp_col+threadIdx.z]
        float tmp = 0;
        if(cur_inp_row >= 0 && cur_inp_row < in_numrow && cur_inp_col >= 0  && cur_inp_col < in_numcol){
            tmp = A[blockIdx.x*in_channel*in_numrow*in_numcol + cur_in_channel*in_numrow*in_numcol + cur_inp_row*in_numcol + cur_inp_col];
        }
        int row_len = in_channel * kernel_size * kernel_size, col_len = out_rownum * out_colnum;
        int idx = blockIdx.x * row_len * col_len + cur_mat_row * col_len + cur_mat_col;
        if(cur_mat_row < row_len && cur_mat_col < col_len) out[idx] = tmp;
    }


    __global__ void matmul_alloc(float* K_mm, float* Inp_mm, float* out, int batch_size, int out_channel, int out_numrow,
                                 int out_numcol, int l1, int l2, int l3){
        // K_mm: l1 x l2, inp_mm: l2 x l3,
        // l1=out_channel, l2=in_channel x kernel_size x kernel_size, l3=out_numrow x out_numcol
        // block: batch x floor(l1 / mm_tile_width) x floor(l3 / mm_tile_width), thread: mm_tile_width x mm_tile_width
        __shared__ float Ks[mm_tilewidth][mm_tilewidth];
        __shared__ float Is[mm_tilewidth][mm_tilewidth];

        int row = blockIdx.y * mm_tilewidth + threadIdx.x;
        int col = blockIdx.z * mm_tilewidth + threadIdx.y;

        float p_value = 0;

        int iter_num = (l2+mm_tilewidth-1)/mm_tilewidth;
        for(int m=0; m<iter_num; m++){
            int read_col = m*mm_tilewidth+threadIdx.y;
            // U[row, m*mm_tilewidth+threadIdx.y]
            if(read_col < l2)
                Ks[threadIdx.x][threadIdx.y] = K_mm[row*l2 + read_col];
            else
                Ks[threadIdx.x][threadIdx.y] = 0;

            int read_row = m*mm_tilewidth+threadIdx.x;
            // V[m*mm_tilewidth+threadIdx.x, col, place_in_36]
            if(read_row < l2)
                Is[threadIdx.x][threadIdx.y] = Inp_mm[blockIdx.x*l2*l3 + read_row*l3 + col];
            else
                Is[threadIdx.x][threadIdx.y] = 0;
            __syncthreads();

            for(int k=0; k<mm_tilewidth; k++){
                p_value += Ks[threadIdx.x][k] * Is[k][threadIdx.y];
            }
            __syncthreads();
        }

        // out[batch_size, cur_out_channel, cur_out_row, cur_out_col]
        if(row < l1 && col < l3){
            // int cur_out_channel = row;
            // int cur_out_row = col / out_numrow, cur_out_col = col % out_numrow;
            int idx = blockIdx.x * l1 * l3 + row * l3 + col;
            out[idx] = p_value;
        }
    }

    __global__ void matmul_alloc_bias(float* K_mm, float* Inp_mm, float* bias, float* out, int batch_size, int out_channel, int out_numrow,
                                      int out_numcol, int l1, int l2, int l3){
        // K_mm: l1 x l2, inp_mm: l2 x l3,
        // l1=out_channel, l2=in_channel x kernel_size x kernel_size, l3=out_numrow x out_numcol
        // block: batch x floor(l1 / mm_tile_width) x floor(l3 / mm_tile_width), thread: mm_tile_width x mm_tile_width
        __shared__ float Ks[mm_tilewidth][mm_tilewidth];
        __shared__ float Is[mm_tilewidth][mm_tilewidth];

        int row = blockIdx.y * mm_tilewidth + threadIdx.y;
        int col = blockIdx.z * mm_tilewidth + threadIdx.x;

        float p_value = 0;

        int iter_num = (l2+mm_tilewidth-1)/mm_tilewidth;
        for(int m=0; m<iter_num; m++){
            int read_col = m*mm_tilewidth+threadIdx.x;
            // U[row, m*mm_tilewidth+threadIdx.y]
            if(read_col < l2)
                Ks[threadIdx.y][threadIdx.x] = K_mm[row*l2 + read_col];
            else
                Ks[threadIdx.y][threadIdx.x] = 0;

            int read_row = m*mm_tilewidth+threadIdx.y;
            // V[m*mm_tilewidth+threadIdx.x, col, place_in_36]
            if(read_row < l2)
                Is[threadIdx.y][threadIdx.x] = Inp_mm[blockIdx.x*l2*l3 + read_row*l3 + col];
            else
                Is[threadIdx.y][threadIdx.x] = 0;
            __syncthreads();
#pragma unroll
            for(int k=0; k<mm_tilewidth; k++){
                p_value += Ks[threadIdx.y][k] * Is[k][threadIdx.x];
            }
            __syncthreads();
        }

        // out[batch_size, cur_out_channel, cur_out_row, cur_out_col]
        if(row < l1 && col < l3){
            // int cur_out_channel = row;
            // int cur_out_row = col / out_numrow, cur_out_col = col % out_numrow;
            int idx = blockIdx.x * l1 * l3 + row * l3 + col;
            out[idx] = p_value + bias[row];
        }
    }

    __global__ void matmul_alloc_bias_relu(float* K_mm, float* Inp_mm, float* bias, float* out, int batch_size, int out_channel, int out_numrow,
                                           int out_numcol, int l1, int l2, int l3){
        // K_mm: l1 x l2, inp_mm: l2 x l3,
        // l1=out_channel, l2=in_channel x kernel_size x kernel_size, l3=out_numrow x out_numcol
        // block: batch x floor(l1 / mm_tile_width) x floor(l3 / mm_tile_width), thread: mm_tile_width x mm_tile_width
        __shared__ float Ks[mm_tilewidth][mm_tilewidth];
        __shared__ float Is[mm_tilewidth][mm_tilewidth];

        int row = blockIdx.y * mm_tilewidth + threadIdx.y;
        int col = blockIdx.z * mm_tilewidth + threadIdx.x;

        float p_value = 0;

        int iter_num = (l2+mm_tilewidth-1)/mm_tilewidth;
        for(int m=0; m<iter_num; m++){
            int read_col = m*mm_tilewidth+threadIdx.x;
            // U[row, m*mm_tilewidth+threadIdx.y]
            if(read_col < l2)
                Ks[threadIdx.y][threadIdx.x] = K_mm[row*l2 + read_col];
            else
                Ks[threadIdx.y][threadIdx.x] = 0;

            int read_row = m*mm_tilewidth+threadIdx.y;
            // V[m*mm_tilewidth+threadIdx.x, col, place_in_36]
            if(read_row < l2)
                Is[threadIdx.y][threadIdx.x] = Inp_mm[blockIdx.x*l2*l3 + read_row*l3 + col];
            else
                Is[threadIdx.y][threadIdx.x] = 0;
            __syncthreads();
#pragma unroll
            for(int k=0; k<mm_tilewidth; k++){
                p_value += Ks[threadIdx.y][k] * Is[k][threadIdx.x];
            }
            __syncthreads();
        }

        // out[batch_size, cur_out_channel, cur_out_row, cur_out_col]
        if(row < l1 && col < l3){
            // int cur_out_channel = row;
            // int cur_out_row = col / out_numrow, cur_out_col = col % out_numrow;
            int idx = blockIdx.x * l1 * l3 + row * l3 + col;
            float tmp = p_value + bias[row];
            out[idx] = (tmp>0)? tmp:0;
        }
    }


    __global__ void matmul_alloc_bias_add_relu(float* K_mm, float* Inp_mm, float* bias, float* out, int batch_size, int out_channel, int out_numrow,
                                           int out_numcol, int l1, int l2, int l3){
        // K_mm: l1 x l2, inp_mm: l2 x l3,
        // l1=out_channel, l2=in_channel x kernel_size x kernel_size, l3=out_numrow x out_numcol
        // block: batch x floor(l1 / mm_tile_width) x floor(l3 / mm_tile_width), thread: mm_tile_width x mm_tile_width
        __shared__ float Ks[mm_tilewidth][mm_tilewidth];
        __shared__ float Is[mm_tilewidth][mm_tilewidth];

        int row = blockIdx.y * mm_tilewidth + threadIdx.y;
        int col = blockIdx.z * mm_tilewidth + threadIdx.x;

        float p_value = 0;

        int iter_num = (l2+mm_tilewidth-1)/mm_tilewidth;
        for(int m=0; m<iter_num; m++){
            int read_col = m*mm_tilewidth+threadIdx.x;
            // U[row, m*mm_tilewidth+threadIdx.y]
            if(read_col < l2)
                Ks[threadIdx.y][threadIdx.x] = K_mm[row*l2 + read_col];
            else
                Ks[threadIdx.y][threadIdx.x] = 0;

            int read_row = m*mm_tilewidth+threadIdx.y;
            // V[m*mm_tilewidth+threadIdx.x, col, place_in_36]
            if(read_row < l2)
                Is[threadIdx.y][threadIdx.x] = Inp_mm[blockIdx.x*l2*l3 + read_row*l3 + col];
            else
                Is[threadIdx.y][threadIdx.x] = 0;
            __syncthreads();
#pragma unroll
            for(int k=0; k<mm_tilewidth; k++){
                p_value += Ks[threadIdx.y][k] * Is[k][threadIdx.x];
            }
            __syncthreads();
        }

        // out[batch_size, cur_out_channel, cur_out_row, cur_out_col]
        if(row < l1 && col < l3){
            // int cur_out_channel = row;
            // int cur_out_row = col / out_numrow, cur_out_col = col % out_numrow;
            int idx = blockIdx.x * l1 * l3 + row * l3 + col;
            float tmp = out[idx] + p_value + bias[row];
            out[idx] = (tmp>0 ? tmp:0);

            // printf("%f\n", tmp);
        }
    }
}