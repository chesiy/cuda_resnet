#include <stdio.h>
#include <iostream>
#include <math.h>

# define mm_tilewidth 8 // tile width when do matrix multiply
// # define load_tile_size 4

// when input is CHW format
__global__ void im2col_CHW(float* A, float* out, int kernel_size, int in_channel, int in_numrow, int in_numcol, 
    int out_rownum, int out_colnum, int stride, int padding, int batch_size){
    // blocks: batch_size x (out_rownum x in_channel) x (K x K), threads: out_colnum
    // out: batch_size x (in_channel * K * K) x (out_rownum * out_colnum)
    // use the strange format in order for Memory Coalescing

    int cur_out_row = blockIdx.y / in_channel, cur_out_col = threadIdx.x, cur_in_channel = blockIdx.y % in_channel;
    int cur_kernel_row = blockIdx.z / kernel_size, cur_kernel_col = blockIdx.z % kernel_size;

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

__global__ void im2col_CHW2(float* A, float* out, int kernel_size, int in_channel, int in_numrow, int in_numcol, 
    int out_rownum, int out_colnum, int stride, int padding, int batch_size){
    // blocks: batch_size x in_channel x (K x K), threads: out_rownum x out_colnum
    // out: batch_size x (in_channel * K * K) x (out_rownum * out_colnum)
    // use the strange format in order for Memory Coalescing

    // int cur_out_row = blockIdx.y / in_channel, cur_out_col = threadIdx.x, cur_in_channel = blockIdx.y % in_channel;
    // int cur_kernel_row = blockIdx.z / kernel_size, cur_kernel_col = blockIdx.z % kernel_size;
    int cur_out_row = threadIdx.x, cur_out_col = threadIdx.y, cur_in_channel = blockIdx.y;
    int cur_kernel_row = blockIdx.z / kernel_size, cur_kernel_col = blockIdx.z % kernel_size;

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
        out[idx] = p_value + bias[row];
    }
}

int main()
{  
    // EASY CASE
    // int in_channels=2, out_channels=1, inp_row=4, inp_col=4, P=1;
    // int batch_size=1;
    // float kernel[18], input[32], output[16]; // kernel: 1*2*3*3, input: 1*2*4*4, output: 1*1*4*4
    // // float U[32], V[16], M[16];
    // for(int i=0; i<18; i++) kernel[i] = i+1;
    // for(int i=0; i<32; i++) input[i] = i+1;

    // float *d_kernel, *d_inp, *d_out;
    // float *d_inp_col;

    // cudaMalloc((void**)&d_kernel, sizeof(float) * 18);
    // cudaMalloc((void**)&d_inp, sizeof(float) * 32);
    // cudaMalloc((void**)&d_out, sizeof(float) * 16);
    // cudaMalloc((void**)&d_inp_col, sizeof(float) * batch_size*18*16);

    // cudaMemcpy(d_kernel, kernel, sizeof(float) * 18, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_inp, input, sizeof(float) * 32, cudaMemcpyHostToDevice);

    // im2col_CHW<<<dim3(1, 8, 9), dim3(4)>>>(d_inp, d_inp_col, 3, in_channels, inp_row, inp_row, 
    //     inp_row, inp_col, 1, 1, 1);
    
    // matmul_alloc<<<dim3(1, 1, 16), dim3(mm_tilewidth, mm_tilewidth)>>>(d_kernel, d_inp_col, d_out, 
    //     batch_size, out_channels, inp_row, inp_col, 1, 18, 16);

    // cudaMemcpy(output, d_out, sizeof(float) * 16, cudaMemcpyDeviceToHost);

    // for(int i=0; i<1; i++){
    //     for(int j=0; j<1; j++){
    //         for(int k=0; k<4; k++){
    //             for(int l=0; l<4; l++){
    //                 float now_element = output[i*16 + j*16 + k*4 + l];
    //                 printf("%f ", now_element);
    //             }
    //             printf(" \n");
    //         }
    //         printf(" \n");
    //     }
    // }


    // HARD CASE
    int in_channels=16, out_channels=8, inp_row=56, inp_col=56, kernel_size=3;
    int batch_size=2, out_row=56, out_col=56;
    float kernel[16*8*3*3], input[2*16*56*56], output[2*8*56*56];
    for(int i=0; i<16*8*3*3; i++) kernel[i] = i;
    for(int i=0; i<2*16*56*56; i++) input[i] = i;
    
    float *d_kernel, *d_inp, *d_out;
    float *d_inp_col;

    cudaMalloc((void**)&d_kernel, sizeof(float) * 16*8*3*3);
    cudaMalloc((void**)&d_inp, sizeof(float) * 2*16*56*56);
    cudaMalloc((void**)&d_out, sizeof(float) * 2*8*56*56);
    cudaMalloc((void**)&d_inp_col, 
        sizeof(float) * batch_size*(in_channels*kernel_size*kernel_size)*(out_row*out_col));

    cudaMemcpy(d_kernel, kernel, sizeof(float) * 16*8*3*3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp, input, sizeof(float) * 2*16*56*56, cudaMemcpyHostToDevice);

    float Onetime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i=0; i<100000;i++){
        im2col_CHW<<<dim3(batch_size, out_row*in_channels, kernel_size*kernel_size), dim3(out_col)>>>
            (d_inp, d_inp_col, kernel_size, in_channels, inp_row, inp_col, out_row, out_col, 1, 1, batch_size);
        // im2col_CHW2<<<dim3(batch_size, in_channels, kernel_size*kernel_size), dim3(out_row, out_col)>>>
        //     (d_inp, d_inp_col, kernel_size, in_channels, inp_row, inp_col, out_row, out_col, 1, 1, batch_size);
        
        matmul_alloc<<<
            dim3(batch_size, (out_channels+mm_tilewidth-1)/mm_tilewidth, (out_row*out_col+mm_tilewidth-1)/mm_tilewidth), 
            dim3(mm_tilewidth, mm_tilewidth)
            >>>(d_kernel, d_inp_col, d_out, batch_size, out_channels, inp_row, inp_col, out_channels, 
            (in_channels*kernel_size*kernel_size), (out_row*out_col));
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Onetime, start, stop);

    cudaMemcpy(output, d_out, sizeof(float) * 2*8*56*56, cudaMemcpyDeviceToHost);
    
    for(int i=0; i<2; i++){
        for(int j=0; j<4; j++){
            for(int k=0; k<7; k++){
                for(int l=0; l<7; l++){
                    float now_element = output[i*196 + j*49 + k*7 + l];
                    printf("%f ", now_element);
                }
                printf(" \n");
            }
            printf(" \n");
        }
    }

    printf("Total Time is: %f\n", Onetime);

    // not divisor
    // int in_channels=2, out_channels=4, inp_row=7, inp_col=7, kernel_size=3;
    // int P=8, batch_size=2, tile_num=4, out_row=7, out_col=7;
    // float kernel[72], input[196], output[392]; // kernel: 4*2*3*3, input: 2*2*7*7, output: 2*4*7*7
    // for(int i=0; i<72; i++) kernel[i] = i;
    // for(int i=0; i<196; i++) input[i] = i;
    
    // float *d_kernel, *d_inp, *d_out;
    // float *d_inp_col;

    // float U[288]; // out_channel(4)*in_channel(2)*36=3

    // cudaMalloc((void**)&d_kernel, sizeof(float) * 72);
    // cudaMalloc((void**)&d_inp, sizeof(float) * 196);
    // cudaMalloc((void**)&d_out, sizeof(float) * 392);
    // cudaMalloc((void**)&d_inp_col, 
    //     sizeof(float) * batch_size*(in_channels*kernel_size*kernel_size)*(out_row*out_col));

    // cudaMemcpy(d_kernel, kernel, sizeof(float) * 72, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_inp, input, sizeof(float) * 196, cudaMemcpyHostToDevice);

    // float Onetime;
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);

    // for(int i=0; i<10000;i++){
    //     im2col_CHW<<<dim3(batch_size, out_row*in_channels, kernel_size*kernel_size), dim3(out_col)>>>
    //         (d_inp, d_inp_col, kernel_size, in_channels, inp_row, inp_col, out_row, out_col, 1, 1, batch_size);
    //     // im2col_CHW2<<<dim3(batch_size, in_channels, kernel_size*kernel_size), dim3(out_row, out_col)>>>
    //     //     (d_inp, d_inp_col, kernel_size, in_channels, inp_row, inp_col, out_row, out_col, 1, 1, batch_size);
        
    //     matmul_alloc<<<
    //         dim3(batch_size, (out_channels+mm_tilewidth-1)/mm_tilewidth, (out_row*out_col+mm_tilewidth-1)/mm_tilewidth), 
    //         dim3(mm_tilewidth, mm_tilewidth)
    //         >>>(d_kernel, d_inp_col, d_out, batch_size, out_channels, inp_row, inp_col, out_channels, 
    //         (in_channels*kernel_size*kernel_size), (out_row*out_col));
    // }

    // cudaDeviceSynchronize();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&Onetime, start, stop);

    // cudaMemcpy(output, d_out, sizeof(float) * 392, cudaMemcpyDeviceToHost);
    
    // for(int i=0; i<2; i++){
    //     for(int j=0; j<4; j++){
    //         for(int k=0; k<7; k++){
    //             for(int l=0; l<7; l++){
    //                 float now_element = output[i*196 + j*49 + k*7 + l];
    //                 printf("%f ", now_element);
    //             }
    //             printf(" \n");
    //         }
    //         printf(" \n");
    //     }
    // }

    // printf("Total Time is: %f\n", Onetime);
    return 0;
}