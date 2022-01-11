#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cuda.h>

namespace matmul{
const int mm_tilewidth = 8;
    __global__ void matmul_bias(float* K_mm, float* Inp_mm, float* bias, float* out, int l1, int l2, int l3){
        // A: l1 x l2, Weight: l3 x l2,
        // l1=out_channel, l2=in_channel x kernel_size x kernel_size, l3=out_numrow x out_numcol
        // block: floor(l1 / mm_tile_width) x floor(l3 / mm_tile_width), thread: mm_tile_width x mm_tile_width
        __shared__ float Ks[mm_tilewidth][mm_tilewidth];
        __shared__ float Is[mm_tilewidth][mm_tilewidth];

        int row = blockIdx.x * mm_tilewidth + threadIdx.y;
        int col = blockIdx.y * mm_tilewidth + threadIdx.x;

        float p_value = 0;

        int iter_num = (l2+mm_tilewidth-1)/mm_tilewidth;
        for(int m=0; m<iter_num; m++){
            int read_col = m*mm_tilewidth+threadIdx.x;
            if(read_col < l2)
                Ks[threadIdx.y][threadIdx.x] = K_mm[row*l2 + read_col];
            else
                Ks[threadIdx.y][threadIdx.x] = 0;

            int read_row = m*mm_tilewidth+threadIdx.y;
            if(read_row < l2)
                // Is[threadIdx.y][threadIdx.x] = Inp_mm[read_row*l3 + col];
                Is[threadIdx.y][threadIdx.x] = Inp_mm[read_row + col*l2];
            else
                Is[threadIdx.y][threadIdx.x] = 0;
            __syncthreads();

    #pragma unroll
            for(int k=0; k<mm_tilewidth; k++){
                p_value += Ks[threadIdx.y][k] * Is[k][threadIdx.x];
            }
            __syncthreads();
        }

        if(row < l1 && col < l3){
            int idx = row * l3 + col;
            out[idx] = p_value + bias[col];
        }
    }
}