#include <stdio.h>
#include <iostream>
#include <math.h>

namespace winograd4{

    const int mm_tilewidth=8; // tile width when do matrix multiply

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

// float G[18] = {
//     0.25, 0, 0,
//     -1.0/6, -1.0/6, -1.0/6,
//     -1.0/6, 1.0/6, -1.0/6,
//     1.0/24, 1.0/12, 1.0/6,
//     1.0/24, -1.0/12, 1.0/6,
//     0, 0, 1
// };

// float G_T[18] = {
//     0.25, -1.0/6, -1.0/6, 1.0/24, 1.0/24, 0,
//     0, -1.0/6, 1.0/6, 1.0/12, -1.0/12, 0,
//     0, -1.0/6, -1.0/6, 1.0/6, 1.0/6, 1
// };

    float G_x24[18] = {
            6, 0, 0,
            -4, -4, -4,
            -4, 4, -4,
            1, 2, 4,
            1, -2, 4,
            0, 0, 24
    };

    float G_T_x24[18] = {
            6, -4, -4, 1, 1, 0,
            0, -4, 4, 2, -2, 0,
            0, -4, -4, 4, 4, 24
    };

    void calc_GgGt(float*g, float*out){
        // G: 6x3, g: 3x3
        float tmp1[18];
        serial_matmul(G_x24, g, tmp1, 6, 3, 3);
        serial_matmul(tmp1, G_T_x24, out, 6, 3, 6);
        for(int i=0; i<36; i++) out[i] = out[i] / 24.0 / 24.0;
    }

    void calc_U(float* kernel, float*U, int in_channels, int out_channels){
        // computed on CPU, no need to be optimized
        for(int k=0; k<out_channels; k++){
            for(int c=0; c<in_channels; c++){
                float* g_kc = kernel + (k*in_channels*9 + c*9); // kernel[k, c]
                float u[36];
                calc_GgGt(g_kc, u);
                for(int i=0; i<36; i++){
                    U[k*in_channels*36 + c*36 + i] = u[i];
                }
            }
        }
    }


    __global__ void calc_V(float* inp, float* V, int P, int batch_size, int in_channels, int in_numrow, int in_numcol, int tile_numrow, int tile_numcol){
        // each block has 36 threads, and in total P*in_channels=(batch_size*tile_num*in_channels) blocks
        __shared__ float inp_shared[6][6];
        __shared__ float Btd_shared[6][6];

        // TODO: OPTIMIZE THIS
        int cur_batch = blockIdx.x, cur_channel = blockIdx.z;
        int cur_row = blockIdx.y / tile_numcol * 4 -1 + threadIdx.x; // tile_row * 4 - 1 + threadIdx.x
        int cur_col = blockIdx.y % tile_numcol * 4 -1 + threadIdx.y; // tile_col * 4 - 1 + threadIdx.y

        if(cur_row >= 0 && cur_row < in_numrow && cur_col >= 0 && cur_col < in_numcol)
            inp_shared[threadIdx.x][threadIdx.y] =
                    inp[cur_batch*in_channels*in_numrow*in_numcol + cur_channel*in_numrow*in_numcol + cur_row*in_numcol + cur_col];
        else
            inp_shared[threadIdx.x][threadIdx.y] = 0;

        __syncthreads();

        // Btd
        switch(threadIdx.x){
            case 0:
                Btd_shared[threadIdx.x][threadIdx.y] = 4*inp_shared[0][threadIdx.y] - 5*inp_shared[2][threadIdx.y] + inp_shared[4][threadIdx.y];
                break;
            case 1:
                Btd_shared[threadIdx.x][threadIdx.y] = -4*inp_shared[1][threadIdx.y] - 4*inp_shared[2][threadIdx.y] +
                                                       inp_shared[3][threadIdx.y] + inp_shared[4][threadIdx.y];
                break;
            case 2:
                Btd_shared[threadIdx.x][threadIdx.y] = 4*inp_shared[1][threadIdx.y] - 4*inp_shared[2][threadIdx.y] -
                                                       inp_shared[3][threadIdx.y] + inp_shared[4][threadIdx.y];
                break;
            case 3:
                Btd_shared[threadIdx.x][threadIdx.y] = -2*inp_shared[1][threadIdx.y] - inp_shared[2][threadIdx.y] +
                                                       2*inp_shared[3][threadIdx.y] + inp_shared[4][threadIdx.y];
                break;
            case 4:
                Btd_shared[threadIdx.x][threadIdx.y] = 2*inp_shared[1][threadIdx.y] - inp_shared[2][threadIdx.y] -
                                                       2*inp_shared[3][threadIdx.y] + inp_shared[4][threadIdx.y];
                break;
            case 5:
                Btd_shared[threadIdx.x][threadIdx.y] = 4*inp_shared[1][threadIdx.y] - 5*inp_shared[3][threadIdx.y] + inp_shared[5][threadIdx.y];
                break;
        }
        // printf("%d %d %d %f %f %f\n", cur_channel, cur_row, cur_col, Btd_shared[threadIdx.x][threadIdx.y], inp_shared[threadIdx.x][threadIdx.y], inp_shared[2][threadIdx.y]);
        __syncthreads();

        // BtdB
        float tmp = 0;
        switch(threadIdx.y){
            case 0:
                tmp = 4*Btd_shared[threadIdx.x][0] - 5*Btd_shared[threadIdx.x][2] + Btd_shared[threadIdx.x][4];
                break;
            case 1:
                tmp = -4*Btd_shared[threadIdx.x][1] - 4*Btd_shared[threadIdx.x][2] + Btd_shared[threadIdx.x][3] + Btd_shared[threadIdx.x][4];
                break;
            case 2:
                tmp = 4*Btd_shared[threadIdx.x][1] - 4*Btd_shared[threadIdx.x][2] - Btd_shared[threadIdx.x][3] + Btd_shared[threadIdx.x][4];
                break;
            case 3:
                tmp = -2*Btd_shared[threadIdx.x][1] - Btd_shared[threadIdx.x][2] + 2*Btd_shared[threadIdx.x][3] + Btd_shared[threadIdx.x][4];
                break;
            case 4:
                tmp = 2*Btd_shared[threadIdx.x][1] - Btd_shared[threadIdx.x][2] - 2*Btd_shared[threadIdx.x][3] + Btd_shared[threadIdx.x][4];
                break;
            case 5:
                tmp = 4*Btd_shared[threadIdx.x][1] - 5*Btd_shared[threadIdx.x][3] + Btd_shared[threadIdx.x][5];
                break;
        }
        // __syncthreads();

        // V[cur_channel, b, threadIdx.x, threadIdx.y] = tmp, and b = (blockIdx.x*tile_numrow*tile_numcol) + blockIdx.y
        V[cur_channel*P*36 + (blockIdx.x*tile_numrow*tile_numcol + blockIdx.y)*36 + threadIdx.x*6 + threadIdx.y] = tmp;

        // printf("%d %d %d %d %f %f %f\n", cur_channel, cur_row, cur_col, cur_channel*P*16 + blockIdx.x*in_numrow*in_numcol*4+blockIdx.y*16 + threadIdx.x*4 + threadIdx.y,
        //     Btd_shared[threadIdx.x][threadIdx.y], inp_shared[threadIdx.x][threadIdx.y], tmp);

        __syncthreads();
    }


    __global__ void calc_UV(float* U, float* V, float* out, int out_channels, int in_channels, int P){
        // U: out_channels x in_channels x 36, V: in_channels x P x 36, out: out_channels x P x 36
        __shared__ float Uds[mm_tilewidth][mm_tilewidth];
        __shared__ float Vds[mm_tilewidth][mm_tilewidth];

        int row = blockIdx.x * mm_tilewidth + threadIdx.x;
        int col = blockIdx.y * mm_tilewidth + threadIdx.y;
        int place_in_36 = blockIdx.z;
        float p_value = 0;

        int iter_num = (in_channels+mm_tilewidth-1)/mm_tilewidth;
        for(int m=0; m<iter_num; m++){
            int read_col = m*mm_tilewidth+threadIdx.y;
            // U[row, m*mm_tilewidth+threadIdx.y, place_in_36]
            if(read_col < in_channels)
                Uds[threadIdx.x][threadIdx.y] = U[row*in_channels*36 + read_col*36 + place_in_36];
            else
                Uds[threadIdx.x][threadIdx.y] = 0;

            int read_row = m*mm_tilewidth+threadIdx.x;
            // V[m*mm_tilewidth+threadIdx.x, col, place_in_36]
            if(read_row < in_channels)
                Vds[threadIdx.x][threadIdx.y] = V[read_row*P*36 + col*36 + place_in_36];
            else
                Vds[threadIdx.x][threadIdx.y] = 0;
            __syncthreads();

            for(int k=0; k<mm_tilewidth; k++){
                p_value += Uds[threadIdx.x][k] * Vds[k][threadIdx.y];
            }
            __syncthreads();
        }

        // printf("%d %d %d %f %f %f \n", row, col, place_in_16, p_value, Uds[threadIdx.x][threadIdx.y], Vds[threadIdx.x][threadIdx.y]);

        if(row < out_channels && col < P)
            out[row*P*36 + col*36 + place_in_36] = p_value;
    }


    __global__ void calc_AtmA(float* M, float* out, int out_channels, int P, int out_numrow, int out_numcol, int tile_num, int tile_numrow, int tile_numcol){
        // each block has 6*6 threads, and in total out_channels*P=(out_channels*batch_size*tile_num) blocks
        // M: out_channels x P * 36
        // TODO: 6*6 threads in a block leads to some inactive threads
        int cur_channel = blockIdx.x, cur_batch = blockIdx.y;
        int cur_tilerow = blockIdx.z / tile_numcol, cur_tilecol = blockIdx.z % tile_numrow;

        // TODO: This memory may be optimized, too; only 6*6 is enough
        __shared__ float m[6][6];
        __shared__ float Atm[4][6];
        m[threadIdx.x][threadIdx.y] = M[cur_channel*P*36 + (cur_batch*tile_num+blockIdx.z)*36 + threadIdx.x*6 + threadIdx.y];
        __syncthreads();

        if(threadIdx.x > 3) return; // valid operation?

        switch(threadIdx.x){
            case 0:
                Atm[threadIdx.x][threadIdx.y] = m[0][threadIdx.y] + m[1][threadIdx.y] + m[2][threadIdx.y] +
                                                m[3][threadIdx.y] + m[4][threadIdx.y];
                break;
            case 1:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] + 2*m[3][threadIdx.y] - 2*m[4][threadIdx.y];
                break;
            case 2:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] + m[2][threadIdx.y] + 4*m[3][threadIdx.y] + 4*m[4][threadIdx.y];
                break;
            case 3:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] + 8*m[3][threadIdx.y] -
                                                8*m[4][threadIdx.y] + m[5][threadIdx.y];
                break;
        }
        __syncthreads();

        if(threadIdx.y > 3) return;

        float tmp = 0;
        switch(threadIdx.y){
            case 0:
                tmp = Atm[threadIdx.x][0] + Atm[threadIdx.x][1] + Atm[threadIdx.x][2] + Atm[threadIdx.x][3] + Atm[threadIdx.x][4];
                break;
            case 1:
                tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] + 2*Atm[threadIdx.x][3] - 2*Atm[threadIdx.x][4];
                break;
            case 2:
                tmp = Atm[threadIdx.x][1] + Atm[threadIdx.x][2] + 4*Atm[threadIdx.x][3] + 4*Atm[threadIdx.x][4];
                break;
            case 3:
                tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] + 8*Atm[threadIdx.x][3] - 8*Atm[threadIdx.x][4] + Atm[threadIdx.x][5];
                break;
        }
        __syncthreads();

        // out[cur_batch, cur_channel, cur_tilerow*4+threadIdx.x, cur_tilecol*4+threadIdx.y]
        int now_row = 4*cur_tilerow+threadIdx.x;
        int now_col = 4*cur_tilecol+threadIdx.y;
        if(now_row < out_numrow && now_col < out_numcol){
            // printf("%d %d \n", now_col, now_row);
            // if(now_col>4 && now_row>4) printf("%d %d %f\n", now_col, now_row, tmp);
            out[cur_batch*out_channels*out_numrow*out_numcol + cur_channel*out_numrow*out_numcol +
                now_row*out_numcol + now_col] = tmp;
        }
    }


    __global__ void calc_AtmA_bias(float* M, float* out, float* bias, int out_channels, int P, int out_numrow, int out_numcol, int tile_num, int tile_numrow, int tile_numcol){
        // each block has 6*6 threads, and in total out_channels*P=(out_channels*batch_size*tile_num) blocks
        // M: out_channels x P * 36
        // TODO: 6*6 threads in a block leads to some inactive threads
        int cur_channel = blockIdx.x, cur_batch = blockIdx.y;
        int cur_tilerow = blockIdx.z / tile_numcol, cur_tilecol = blockIdx.z % tile_numrow;

        // TODO: This memory may be optimized, too; only 6*6 is enough
        __shared__ float m[6][6];
        __shared__ float Atm[4][6];
        m[threadIdx.x][threadIdx.y] = M[cur_channel*P*36 + (cur_batch*tile_num+blockIdx.z)*36 + threadIdx.x*6 + threadIdx.y];
        __syncthreads();

        if(threadIdx.x > 3) return; // valid operation?

        switch(threadIdx.x){
            case 0:
                Atm[threadIdx.x][threadIdx.y] = m[0][threadIdx.y] + m[1][threadIdx.y] + m[2][threadIdx.y] +
                                                m[3][threadIdx.y] + m[4][threadIdx.y];
                break;
            case 1:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] + 2*m[3][threadIdx.y] - 2*m[4][threadIdx.y];
                break;
            case 2:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] + m[2][threadIdx.y] + 4*m[3][threadIdx.y] + 4*m[4][threadIdx.y];
                break;
            case 3:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] + 8*m[3][threadIdx.y] -
                                                8*m[4][threadIdx.y] + m[5][threadIdx.y];
                break;
        }
        __syncthreads();

        if(threadIdx.y > 3) return;

        float tmp = 0;
        switch(threadIdx.y){
            case 0:
                tmp = Atm[threadIdx.x][0] + Atm[threadIdx.x][1] + Atm[threadIdx.x][2] + Atm[threadIdx.x][3] + Atm[threadIdx.x][4];
                break;
            case 1:
                tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] + 2*Atm[threadIdx.x][3] - 2*Atm[threadIdx.x][4];
                break;
            case 2:
                tmp = Atm[threadIdx.x][1] + Atm[threadIdx.x][2] + 4*Atm[threadIdx.x][3] + 4*Atm[threadIdx.x][4];
                break;
            case 3:
                tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] + 8*Atm[threadIdx.x][3] - 8*Atm[threadIdx.x][4] + Atm[threadIdx.x][5];
                break;
        }
        // __syncthreads();

        // out[cur_batch, cur_channel, cur_tilerow*4+threadIdx.x, cur_tilecol*4+threadIdx.y]
        int now_row = 4*cur_tilerow+threadIdx.x;
        int now_col = 4*cur_tilecol+threadIdx.y;
        if(now_row < out_numrow && now_col < out_numcol)
            out[cur_batch*out_channels*out_numrow*out_numcol + cur_channel*out_numrow*out_numcol +
                now_row*out_numcol + now_col] = tmp + bias[cur_channel];
    }

    __global__ void calc_AtmA_bias_relu(float* M, float* out, float* bias, int out_channels, int P, int out_numrow, int out_numcol, int tile_num, int tile_numrow, int tile_numcol){
        // each block has 6*6 threads, and in total out_channels*P=(out_channels*batch_size*tile_num) blocks
        // M: out_channels x P * 36
        // TODO: 6*6 threads in a block leads to some inactive threads
        int cur_channel = blockIdx.x, cur_batch = blockIdx.y;
        int cur_tilerow = blockIdx.z / tile_numcol, cur_tilecol = blockIdx.z % tile_numrow;

        // TODO: This memory may be optimized, too; only 6*6 is enough
        __shared__ float m[6][6];
        __shared__ float Atm[4][6];
        m[threadIdx.x][threadIdx.y] = M[cur_channel*P*36 + (cur_batch*tile_num+blockIdx.z)*36 + threadIdx.x*6 + threadIdx.y];
        __syncthreads();

        if(threadIdx.x > 3) return; // valid operation?

        switch(threadIdx.x){
            case 0:
                Atm[threadIdx.x][threadIdx.y] = m[0][threadIdx.y] + m[1][threadIdx.y] + m[2][threadIdx.y] +
                                                m[3][threadIdx.y] + m[4][threadIdx.y];
                break;
            case 1:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] + 2*m[3][threadIdx.y] - 2*m[4][threadIdx.y];
                break;
            case 2:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] + m[2][threadIdx.y] + 4*m[3][threadIdx.y] + 4*m[4][threadIdx.y];
                break;
            case 3:
                Atm[threadIdx.x][threadIdx.y] = m[1][threadIdx.y] - m[2][threadIdx.y] + 8*m[3][threadIdx.y] -
                                                8*m[4][threadIdx.y] + m[5][threadIdx.y];
                break;
        }
        __syncthreads();

        if(threadIdx.y > 3) return;

        float tmp = 0;
        switch(threadIdx.y){
            case 0:
                tmp = Atm[threadIdx.x][0] + Atm[threadIdx.x][1] + Atm[threadIdx.x][2] + Atm[threadIdx.x][3] + Atm[threadIdx.x][4];
                break;
            case 1:
                tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] + 2*Atm[threadIdx.x][3] - 2*Atm[threadIdx.x][4];
                break;
            case 2:
                tmp = Atm[threadIdx.x][1] + Atm[threadIdx.x][2] + 4*Atm[threadIdx.x][3] + 4*Atm[threadIdx.x][4];
                break;
            case 3:
                tmp = Atm[threadIdx.x][1] - Atm[threadIdx.x][2] + 8*Atm[threadIdx.x][3] - 8*Atm[threadIdx.x][4] + Atm[threadIdx.x][5];
                break;
        }
        // __syncthreads();

        // out[cur_batch, cur_channel, cur_tilerow*4+threadIdx.x, cur_tilecol*4+threadIdx.y]
        int now_row = 4*cur_tilerow+threadIdx.x;
        int now_col = 4*cur_tilecol+threadIdx.y;
        if(now_row < out_numrow && now_col < out_numcol)
            out[cur_batch*out_channels*out_numrow*out_numcol + cur_channel*out_numrow*out_numcol +
                now_row*out_numcol + now_col] = ((tmp + bias[cur_channel])>0)?(tmp + bias[cur_channel]):0;
    }


    __global__ void print_device(float* M, int length){
        for(int i=0; i<length; i++) printf("%f ", M[i]);
    }

}



//int main()
//{
//    // EASY CASE
//    // int in_channels=2, out_channels=1, inp_row=4, inp_col=4, P=1;
//    // float kernel[18], input[32], output[16]; // kernel: 1*2*3*3, input: 1*2*4*4, output: 1*1*4*4
//    // // float U[32], V[16], M[16];
//    // for(int i=0; i<18; i++) kernel[i] = i+1;
//    // for(int i=0; i<32; i++) input[i] = i+1;
//
//    // float *d_kernel, *d_inp, *d_out;
//    // float *d_V, *d_U, *d_UV;
//
//    // float U[72]; // out_channel(1)*in_channel(2)*36
//
//    // cudaMalloc((void**)&d_kernel, sizeof(float) * 18);
//    // cudaMalloc((void**)&d_inp, sizeof(float) * 32);
//    // cudaMalloc((void**)&d_out, sizeof(float) * 16);
//
//    // cudaMemcpy(d_kernel, kernel, sizeof(float) * 18, cudaMemcpyHostToDevice);
//    // cudaMemcpy(d_inp, input, sizeof(float) * 32, cudaMemcpyHostToDevice);
//
//    // cudaMalloc((void**)&d_V, sizeof(float) * in_channels*P*36);
//    // cudaMalloc((void**)&d_U, sizeof(float) * out_channels*in_channels*36);
//    // cudaMalloc((void**)&d_UV, sizeof(float) * out_channels*P*36);
//
//    // calc_U(kernel, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand
//
//    // cudaMemcpy(d_U, U, sizeof(float) * 72, cudaMemcpyHostToDevice);
//
//    // calc_V<<<dim3(1, 1, in_channels), dim3(6, 6)>>>(d_inp, d_V, P, 1, in_channels, inp_row, inp_col);
//
//    // // print_device<<<dim3(1), dim3(1)>>>(d_V, in_channels*P*36);
//
//    // calc_UV<<<dim3(1, 1, 36), dim3(1, 1)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
//    // calc_AtmA<<<dim3(out_channels, 1, 1), dim3(6, 6)>>>(d_UV, d_out, out_channels, P, inp_row, inp_col, 1);
//
//    // cudaMemcpy(output, d_out, sizeof(float) * 16, cudaMemcpyDeviceToHost);
//
//    // for(int i=0; i<1; i++){
//    //     for(int j=0; j<1; j++){
//    //         for(int k=0; k<4; k++){
//    //             for(int l=0; l<4; l++){
//    //                 float now_element = output[i*16 + j*16 + k*4 + l];
//    //                 printf("%f ", now_element);
//    //             }
//    //             printf(" \n");
//    //         }
//    //         printf(" \n");
//    //     }
//    // }
//
//
//    // HARD CASE
//    int in_channels=2, out_channels=4, inp_row=8, inp_col=8;
//    int P=8, batch_size=2, tile_num=4, out_row=8, out_col=8;
//    float kernel[72], input[256], output[512]; // kernel: 4*2*3*3, input: 2*2*8*8, output: 2*4*8*8
//    for(int i=0; i<72; i++) kernel[i] = i;
//    for(int i=0; i<256; i++) input[i] = i;
//
//    float *d_kernel, *d_inp, *d_out;
//    float *d_V, *d_U, *d_UV;
//
//    float U[288]; // out_channel(4)*in_channel(2)*36
//
//    cudaMalloc((void**)&d_kernel, sizeof(float) * 72);
//    cudaMalloc((void**)&d_inp, sizeof(float) * 256);
//    cudaMalloc((void**)&d_out, sizeof(float) * 512);
//
//    cudaMemcpy(d_kernel, kernel, sizeof(float) * 72, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_inp, input, sizeof(float) * 256, cudaMemcpyHostToDevice);
//
//    cudaMalloc((void**)&d_V, sizeof(float) * in_channels*P*36);
//    cudaMalloc((void**)&d_U, sizeof(float) * out_channels*in_channels*36);
//    cudaMalloc((void**)&d_UV, sizeof(float) * out_channels*P*36);
//
//    calc_U(kernel, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand
//
//    cudaMemcpy(d_U, U, sizeof(float) * 288, cudaMemcpyHostToDevice);
//
//    calc_V<<<dim3(batch_size, tile_num, in_channels), dim3(6, 6)>>>(d_inp, d_V, P, batch_size, in_channels, inp_row, inp_col);
//    calc_UV<<<dim3(out_channels/2, P/2, 36), dim3(2, 2)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
//    calc_AtmA<<<dim3(out_channels, batch_size, tile_num), dim3(6, 6)>>>(d_UV, d_out, out_channels, P, out_row, out_col, tile_num);
//
//    cudaMemcpy(output, d_out, sizeof(float) * 512, cudaMemcpyDeviceToHost);
//
//    for(int i=0; i<2; i++){
//        for(int j=0; j<4; j++){
//            for(int k=0; k<8; k++){
//                for(int l=0; l<8; l++){
//                    float now_element = output[i*256 + j*64 + k*8 + l];
//                    printf("%f ", now_element);
//                }
//                printf(" \n");
//            }
//            printf(" \n");
//        }
//    }
//    return 0;
//}