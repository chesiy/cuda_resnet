#include <stdio.h>
#include <iostream>
#include <math.h>

# define mm_tilewidth 7 // tile width when do matrix multiply

void serial_matmul(float* A0, float*B0, float*C0, 
    int dim_1, int dim_2, int dim_3){
    // A: dim1 x dim2, B: dim2 x dim3, C: dim1 x dim3
    for(int i=0; i<dim_1; i++){
        for(int j=0; j<dim_3; j++){
            float tmp = 0;
            for(int k=0; k<dim_2; k++){
                tmp += A0[i*dim_2+k] * B0[k*dim_3+j];
            }
            C0[i*dim_3+j] = tmp;
        }
    }
}

float G[12] = {
    1, 0, 0,
    0.5, 0.5, 0.5,
    0.5, -0.5, 0.5,
    0, 0, 1
};

float G_T[12] = {
    1, 0.5, 0.5, 0,
    0, 0.5, -0.5, 0,
    0, 0.5, 0.5, 1
};

void calc_GgGt(float*g, float*out){
    // G: 4x3, g: 3x3
    float tmp1[12];
    serial_matmul(G, g, tmp1, 4, 3, 3);
    serial_matmul(tmp1, G_T, out, 4, 3, 4);
}

void calc_U(float* kernel, float*U, int in_channels, int out_channels){
    // computed on CPU, no need to be optimized
    for(int k=0; k<out_channels; k++){
        for(int c=0; c<in_channels; c++){
            float* g_kc = kernel + (k*in_channels*9 + c*9); // kernel[k, c]
            float u[16];
            calc_GgGt(g_kc, u);
            for(int i=0; i<16; i++){
                U[k*in_channels*16 + c*16 + i] = u[i];
            }
        }
    }
}


__global__ void calc_V(float* inp, float* V, int P, int batch_size, int in_channels, int in_numrow, int in_numcol, int tile_numrow, int tile_numcol){
    // each block has 16 threads, and in total P*in_channels=(batch_size*tile_num*in_channels) blocks
    __shared__ float inp_shared[4][4];
    __shared__ float Btd_shared[4][4];

    // TODO: OPTIMIZE THIS
    int cur_batch = blockIdx.x, cur_channel = blockIdx.z;
    int cur_row = blockIdx.y / tile_numcol * 2 -1 + threadIdx.x; // tile_row * 2 - 1 + threadIdx.x
    int cur_col = blockIdx.y % tile_numcol * 2 -1 + threadIdx.y; // tile_col * 2 - 1 + threadIdx.y

    if(cur_row >= 0 && cur_row < in_numrow && cur_col >= 0 && cur_col < in_numcol) 
        inp_shared[threadIdx.x][threadIdx.y] = 
            inp[cur_batch*in_channels*in_numrow*in_numcol + cur_channel*in_numrow*in_numcol + cur_row*in_numcol + cur_col];
    else
        inp_shared[threadIdx.x][threadIdx.y] = 0;

    __syncthreads();

    // Btd
    switch(threadIdx.x){
        case 0:
            Btd_shared[threadIdx.x][threadIdx.y] = inp_shared[0][threadIdx.y] - inp_shared[2][threadIdx.y];
            break;
            // printf("%d %d %d %f %f %f\n", cur_channel, threadIdx.x, threadIdx.y, Btd_shared[threadIdx.x][threadIdx.y], inp_shared[0][threadIdx.y], inp_shared[2][threadIdx.y]);
        case 1:
            Btd_shared[threadIdx.x][threadIdx.y] = inp_shared[1][threadIdx.y] + inp_shared[2][threadIdx.y];
            break;
        case 2:
            Btd_shared[threadIdx.x][threadIdx.y] = -inp_shared[1][threadIdx.y] + inp_shared[2][threadIdx.y];
            break;
        case 3:
            Btd_shared[threadIdx.x][threadIdx.y] = inp_shared[1][threadIdx.y] - inp_shared[3][threadIdx.y];
            break;
    }
    // printf("%d %d %d %f %f %f\n", cur_channel, cur_row, cur_col, Btd_shared[threadIdx.x][threadIdx.y], inp_shared[threadIdx.x][threadIdx.y], inp_shared[2][threadIdx.y]);
    __syncthreads();

    // BtdB
    float tmp = 0;
    switch(threadIdx.y){
        case 0:
            tmp = Btd_shared[threadIdx.x][0] - Btd_shared[threadIdx.x][2];
            break;
        case 1:
            tmp = Btd_shared[threadIdx.x][1] + Btd_shared[threadIdx.x][2];
            break;
        case 2:
            tmp = -Btd_shared[threadIdx.x][1] + Btd_shared[threadIdx.x][2];
            break;
        case 3:
            tmp = Btd_shared[threadIdx.x][1] - Btd_shared[threadIdx.x][3];
            break;
    }
    // __syncthreads();

    // V[cur_channel, b, threadIdx.x, threadIdx.y] = tmp, and b = (blockIdx.x*tile_numrow*tile_numcol) + blockIdx.y
    V[cur_channel*P*16 + (blockIdx.x*tile_numrow*tile_numcol + blockIdx.y)*16 + threadIdx.x*4 + threadIdx.y] = tmp;

    // printf("%d %d %d %d %f %f %f\n", cur_channel, cur_row, cur_col, cur_channel*P*16 + blockIdx.x*in_numrow*in_numcol*4+blockIdx.y*16 + threadIdx.x*4 + threadIdx.y, 
    //     Btd_shared[threadIdx.x][threadIdx.y], inp_shared[threadIdx.x][threadIdx.y], tmp);

    __syncthreads(); // ? before call next function, many synch simultaneously ?
}


__global__ void calc_UV(float* U, float* V, float* out, int out_channels, int in_channels, int P){
    // U: out_channels x in_channels x 16, V: in_channels x P x 16, out: out_channels x P x 16
    __shared__ float Uds[mm_tilewidth][mm_tilewidth];
    __shared__ float Vds[mm_tilewidth][mm_tilewidth];
    
    int row = blockIdx.x * mm_tilewidth + threadIdx.x;
    int col = blockIdx.y * mm_tilewidth + threadIdx.y;
    int place_in_16 = blockIdx.z;
    float p_value = 0;

    int iter_num = (in_channels+mm_tilewidth-1)/mm_tilewidth;
    for(int m=0; m<iter_num; m++){
        // U[row, m*mm_tilewidth+threadIdx.y, place_in_16]
        int read_col = m*mm_tilewidth+threadIdx.y;
        if(read_col < in_channels)
            Uds[threadIdx.x][threadIdx.y] = U[row*in_channels*16 + read_col*16 + place_in_16]; 
        else
            Uds[threadIdx.x][threadIdx.y] = 0;
        
        int read_row = m*mm_tilewidth+threadIdx.x;
        // V[m*mm_tilewidth+threadIdx.x, col, place_in_16]
        if(read_row < in_channels)
            Vds[threadIdx.x][threadIdx.y] = V[read_row*P*16 + col*16 + place_in_16];
        else
            Vds[threadIdx.x][threadIdx.y] = 0;
        
        // printf("%d %d\n", read_col, read_row);
        __syncthreads();

        for(int k=0; k<mm_tilewidth; k++){
            p_value += Uds[threadIdx.x][k] * Vds[k][threadIdx.y];
        }
        __syncthreads();
    }

    // printf("%d %d %d %f %f %f \n", row, col, place_in_16, p_value, Uds[threadIdx.x][threadIdx.y], Vds[threadIdx.x][threadIdx.y]);
    if(row < out_channels && col < P)
        out[row*P*16 + col*16 + place_in_16] = p_value;
}


__global__ void calc_AtmA(float* M, float* out, int out_channels, int P, int out_numrow, int out_numcol, int tile_num, int tile_numrow, int tile_numcol){
    // each block has 4 threads, and in total out_channels*P=(out_channels*batch_size*tile_num) blocks
    // M: out_channels x P * 16
    int cur_channel = blockIdx.x, cur_batch = blockIdx.y;
    int cur_tilerow = blockIdx.z / tile_numcol, cur_tilecol = blockIdx.z % tile_numrow;

    __shared__ float m[4][4];
    __shared__ float Atm[2][4];
    m[threadIdx.x][threadIdx.y] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + threadIdx.x*4 + threadIdx.y];
    m[threadIdx.x][threadIdx.y+2] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + threadIdx.x*4 + threadIdx.y + 2];
    m[threadIdx.x+2][threadIdx.y] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + (threadIdx.x+2)*4 + threadIdx.y];
    m[threadIdx.x+2][threadIdx.y+2] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + (threadIdx.x+2)*4 + threadIdx.y + 2];
    __syncthreads();

    switch(threadIdx.x){
        case 0:
            Atm[threadIdx.x][threadIdx.y] = m[0][threadIdx.y] + m[1][threadIdx.y] + m[2][threadIdx.y];
            Atm[threadIdx.x][threadIdx.y+2] = m[0][threadIdx.y+2] + m[1][threadIdx.y+2] + m[2][threadIdx.y+2];
            break;
        case 1:
            Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] - m[3][threadIdx.y];
            Atm[threadIdx.x][threadIdx.y+2] = m[1][threadIdx.y+2] - m[2][threadIdx.y+2] - m[3][threadIdx.y+2];
            break;
    }
    __syncthreads();

    float tmp = 0;
    switch(threadIdx.y){
        case 0:
            tmp = Atm[threadIdx.x][0] + Atm[threadIdx.x][1] + Atm[threadIdx.x][2];
            break;
        case 1:
            tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] - Atm[threadIdx.x][3];
            break;
    }
    // __syncthreads();

    // out[cur_batch, cur_channel, cur_tilerow*2+threadIdx.x, cur_tilecol*2+threadIdx.y]
    int now_row = 2*cur_tilerow+threadIdx.x;
    int now_col = 2*cur_tilecol+threadIdx.y;
    if(now_row < out_numrow && now_col < out_numcol){
        out[cur_batch*out_channels*out_numrow*out_numcol + cur_channel*out_numrow*out_numcol + 
                now_row*out_numcol + now_col] = tmp; 
    }
}


__global__ void calc_AtmA_bias(float* M, float* out, float* bias, int out_channels, int P, int out_numrow, int out_numcol, int tile_num, int tile_numrow, int tile_numcol){
    // each block has 4 threads, and in total out_channels*P=(out_channels*batch_size*tile_num) blocks
    // M: out_channels x P * 16
    int cur_channel = blockIdx.x, cur_batch = blockIdx.y;
    int cur_tilerow = blockIdx.z / tile_numcol, cur_tilecol = blockIdx.z % tile_numrow;

    __shared__ float m[4][4];
    __shared__ float Atm[2][4];
    m[threadIdx.x][threadIdx.y] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + threadIdx.x*4 + threadIdx.y];
    m[threadIdx.x][threadIdx.y+2] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + threadIdx.x*4 + threadIdx.y + 2];
    m[threadIdx.x+2][threadIdx.y] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + (threadIdx.x+2)*4 + threadIdx.y];
    m[threadIdx.x+2][threadIdx.y+2] = M[cur_channel*P*16 + (cur_batch*tile_num+blockIdx.z)*16 + (threadIdx.x+2)*4 + threadIdx.y + 2];
    __syncthreads();

    switch(threadIdx.x){
        case 0:
            Atm[threadIdx.x][threadIdx.y] = m[0][threadIdx.y] + m[1][threadIdx.y] + m[2][threadIdx.y];
            Atm[threadIdx.x][threadIdx.y+2] = m[0][threadIdx.y+2] + m[1][threadIdx.y+2] + m[2][threadIdx.y+2];
            break;
        case 1:
            Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] - m[3][threadIdx.y];
            Atm[threadIdx.x][threadIdx.y+2] = m[1][threadIdx.y+2] - m[2][threadIdx.y+2] - m[3][threadIdx.y+2];
            break;
    }
    __syncthreads();

    float tmp = 0;
    switch(threadIdx.y){
        case 0:
            tmp = Atm[threadIdx.x][0] + Atm[threadIdx.x][1] + Atm[threadIdx.x][2];
            break;
        case 1:
            tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] - Atm[threadIdx.x][3];
            break;
    }

    // out[cur_batch, cur_channel, cur_tilerow*2+threadIdx.x, cur_tilecol*2+threadIdx.y]
    int now_row = 2*cur_tilerow+threadIdx.x;
    int now_col = 2*cur_tilecol+threadIdx.y;
    if(now_row < out_numrow && now_col < out_numcol){
        out[cur_batch*out_channels*out_numrow*out_numcol + cur_channel*out_numrow*out_numcol + 
                now_row*out_numcol + now_col] = tmp + bias[cur_channel]; 
    }
    // __syncthreads();
}


__global__ void print_device(float* M, int length){
    for(int i=0; i<length; i++) printf("%f ", M[i]);
}


int main()
{  
    // EASY CASE
    // int in_channels=2, out_channels=1, inp_row=2, inp_col=2, P=1;
    // float kernel[18], input[8], output[4]; // kernel: 1*2*3*3, input: 1*2*2*2, output: 1*1*2*2
    // // float U[32], V[16], M[16];
    // for(int i=0; i<18; i++) kernel[i] = i+1;
    // for(int i=0; i<8; i++) input[i] = i+1;

    // float *d_kernel, *d_inp, *d_out;
    // float *d_V, *d_U, *d_UV;

    // float U[32]; // out_channel(1)*in_channel(2)*16

    // cudaMalloc((void**)&d_kernel, sizeof(float) * 18);
    // cudaMalloc((void**)&d_inp, sizeof(float) * 8);
    // cudaMalloc((void**)&d_out, sizeof(float) * 4);

    // cudaMemcpy(d_kernel, kernel, sizeof(float) * 18, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_inp, input, sizeof(float) * 8, cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&d_V, sizeof(float) * in_channels*P*16);
    // cudaMalloc((void**)&d_U, sizeof(float) * out_channels*in_channels*16);
    // cudaMalloc((void**)&d_UV, sizeof(float) * out_channels*P*16);

    // calc_U(kernel, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

    // cudaMemcpy(d_U, U, sizeof(float) * 32, cudaMemcpyHostToDevice);

    // calc_V<<<dim3(1, 1, in_channels), dim3(4, 4)>>>(d_inp, d_V, P, 1, in_channels, inp_row, inp_col);
    // calc_UV<<<dim3(1, 1, 16), dim3(1, 1)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
    // calc_AtmA<<<dim3(out_channels, 1, 1), dim3(2, 2)>>>(d_UV, d_out, out_channels, P, 2, 2, 1);

    // cudaMemcpy(output, d_out, sizeof(float) * 4, cudaMemcpyDeviceToHost);

    // for(int i=0; i<1; i++){
    //     for(int j=0; j<1; j++){
    //         for(int k=0; k<2; k++){
    //             for(int l=0; l<2; l++){
    //                 float now_element = output[i*4 + j*4 + k*2 + l];
    //                 printf("%f ", now_element);
    //             }
    //             printf(" \n");
    //         }
    //         printf(" \n");
    //     }
    // }

    // int in_channels=2, out_channels=1, inp_row=4, inp_col=4, P=4;
    // float kernel[18], input[32], output[16]; // kernel: 1*2*3*3, input: 1*2*4*4, output: 1*1*4*4
    // // float U[32], V[16], M[16];
    // for(int i=0; i<18; i++) kernel[i] = i+1;
    // for(int i=0; i<32; i++) input[i] = i+1;

    // float *d_kernel, *d_inp, *d_out;
    // float *d_V, *d_U, *d_UV;

    // float U[32]; // out_channel(1)*in_channel(2)*16

    // cudaMalloc((void**)&d_kernel, sizeof(float) * 18);
    // cudaMalloc((void**)&d_inp, sizeof(float) * 32);
    // cudaMalloc((void**)&d_out, sizeof(float) * 16);

    // cudaMemcpy(d_kernel, kernel, sizeof(float) * 18, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_inp, input, sizeof(float) * 32, cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&d_V, sizeof(float) * in_channels*P*16);
    // cudaMalloc((void**)&d_U, sizeof(float) * out_channels*in_channels*16);
    // cudaMalloc((void**)&d_UV, sizeof(float) * out_channels*P*16);

    // calc_U(kernel, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

    // cudaMemcpy(d_U, U, sizeof(float) * 32, cudaMemcpyHostToDevice);

    // calc_V<<<dim3(1, 4, in_channels), dim3(4, 4)>>>(d_inp, d_V, P, 1, in_channels, inp_row, inp_col);

    // print_device<<<dim3(1), dim3(1)>>>(d_V, in_channels*P*16);

    // calc_UV<<<dim3(1, 4, 16), dim3(1, 1)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
    // calc_AtmA<<<dim3(out_channels, 1, 4), dim3(2, 2)>>>(d_UV, d_out, out_channels, P, 4, 4, 4);

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
    // int in_channels=2, out_channels=4, inp_row=6, inp_col=6;
    // int P=18, batch_size=2, tile_num=9, out_row=6, out_col=6;
    // float kernel[72], input[144], output[288]; // kernel: 4*2*3*3, input: 2*2*6*6, output: 2*4*6*6
    // for(int i=0; i<72; i++) kernel[i] = i;
    // for(int i=0; i<144; i++) input[i] = i;
    
    // float *d_kernel, *d_inp, *d_out;
    // float *d_V, *d_U, *d_UV;

    // float U[128]; // out_channel(4)*in_channel(2)*16

    // cudaMalloc((void**)&d_kernel, sizeof(float) * 72);
    // cudaMalloc((void**)&d_inp, sizeof(float) * 144);
    // cudaMalloc((void**)&d_out, sizeof(float) * 288);

    // cudaMemcpy(d_kernel, kernel, sizeof(float) * 72, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_inp, input, sizeof(float) * 144, cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&d_V, sizeof(float) * in_channels*P*16);
    // cudaMalloc((void**)&d_U, sizeof(float) * out_channels*in_channels*16);
    // cudaMalloc((void**)&d_UV, sizeof(float) * out_channels*P*16);

    // calc_U(kernel, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

    // cudaMemcpy(d_U, U, sizeof(float) * 128, cudaMemcpyHostToDevice);

    // calc_V<<<dim3(batch_size, tile_num, in_channels), dim3(4, 4)>>>(d_inp, d_V, P, batch_size, in_channels, inp_row, inp_col);
    // calc_UV<<<dim3(out_channels/2, P/2, 16), dim3(2, 2)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
    // calc_AtmA<<<dim3(out_channels, batch_size, tile_num), dim3(2, 2)>>>(d_UV, d_out, out_channels, P, out_row, out_col, tile_num);

    // cudaMemcpy(output, d_out, sizeof(float) * 288, cudaMemcpyDeviceToHost);
    
    // for(int i=0; i<2; i++){
    //     for(int j=0; j<4; j++){
    //         for(int k=0; k<6; k++){
    //             for(int l=0; l<6; l++){
    //                 float now_element = output[i*144 + j*36 + k*6 + l];
    //                 printf("%f ", now_element);
    //             }
    //             printf(" \n");
    //         }
    //         printf(" \n");
    //     }
    // }
    // return 0;

    int in_channels=2, out_channels=4, inp_row=7, inp_col=7;
    int P=32, batch_size=2, tile_num=16, out_row=7, out_col=7;
    float kernel[72], input[196], output[392]; // kernel: 4*2*3*3, input: 2*2*7*7, output: 2*4*7*7
    for(int i=0; i<72; i++) kernel[i] = i;
    for(int i=0; i<196; i++) input[i] = i;
    
    float *d_kernel, *d_inp, *d_out;
    float *d_V, *d_U, *d_UV;

    float U[288]; // out_channel(4)*in_channel(2)*36

    cudaMalloc((void**)&d_kernel, sizeof(float) * 72);
    cudaMalloc((void**)&d_inp, sizeof(float) * 196);
    cudaMalloc((void**)&d_out, sizeof(float) * 392);

    cudaMemcpy(d_kernel, kernel, sizeof(float) * 72, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp, input, sizeof(float) * 196, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_V, sizeof(float) * in_channels*P*16);
    cudaMalloc((void**)&d_U, sizeof(float) * out_channels*in_channels*16);
    cudaMalloc((void**)&d_UV, sizeof(float) * out_channels*P*16);

    calc_U(kernel, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

    cudaMemcpy(d_U, U, sizeof(float) * 128, cudaMemcpyHostToDevice);

    calc_V<<<dim3(batch_size, tile_num, in_channels), dim3(4, 4)>>>(d_inp, d_V, P, batch_size, in_channels, inp_row, inp_col, 4, 4);
    calc_UV<<<dim3((out_channels+6)/7, (P+6)/7, 16), dim3(7, 7)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
    calc_AtmA<<<dim3(out_channels, batch_size, tile_num), dim3(2, 2)>>>(d_UV, d_out, out_channels, P, out_row, out_col, tile_num, 4, 4);

    cudaMemcpy(output, d_out, sizeof(float) * 392, cudaMemcpyDeviceToHost);
    
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
    return 0;
}