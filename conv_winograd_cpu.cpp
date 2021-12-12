#include <stdio.h>
#include <iostream>
#include <math.h>

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

float B_T[16] = {
    1, 0, -1, 0,
    0, 1, 1, 0,
    0, -1, 1, 0,
    0, 1, 0, -1
};
    
float B[16] = {
    1, 0, 0, 0,
    0, 1, -1, 1, 
    -1, 1, 1, 0,
    0, 0, 0, -1
};

float A_T[8] = {
    1, 1, 1, 0,
    0, 1, -1, -1
};

float A[8] = {
    1, 0,
    1, 1, 
    1, -1,
    0, -1
};


void calc_GgGt(float*g, float*out){
    // G: 4x3, g: 3x3
    float tmp1[12];
    serial_matmul(G, g, tmp1, 4, 3, 3);
    serial_matmul(tmp1, G_T, out, 4, 3, 4);
}

void calc_BtdB(float*d, float*out){
    // B: 4x4
    float tmp1[16];
    serial_matmul(B_T, d, tmp1, 4, 4, 4);
    serial_matmul(tmp1, B, out, 4, 4, 4);
    return;
}

void calc_AtmA(float*m, float*out){
    float tmp1[8];
    serial_matmul(A_T, m, tmp1, 2, 4, 4);
    serial_matmul(tmp1, A, out, 2, 4, 2);
    return;
}


void matmul_4x4(float* U, float* V, float* M, int out_channels, int in_channels, int P){
    // U: out_channnels x in channels x 16, V: in_channels x P x 16, M: out_channels x P x 16
    for(int t=0; t<16; t++){
        for(int i=0; i<out_channels; i++){
            for(int j=0; j<P; j++){
                float tmp = 0;
                for(int k=0; k<in_channels; k++){
                    tmp += U[i*in_channels*16 + k*16 + t] * V[k*P*16 + j*16 + t];
                }
                M[i*P*16 + j*16 + t] = tmp;
                // printf("%f ", tmp);
            }
        }
    }
}

void batch_trivial_conv2d_2x2_3x3(float* A_b, float*kernel, float*C_b,
    int batch_size, int in_numrow, int in_numcol, int in_channels, int out_channels,
    int padding){
    // padding=1, stride=1
    // each thread calulates 4x4, 3x3
    int out_numrow = (in_numrow + padding*2 - 3) + 1;
    int out_numcol = (in_numcol + padding*2 - 3) + 1;

    float* U = new float[out_channels*in_channels*16]; // K x C x 16
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

    // for(int i=0; i<out_channels*in_channels*16; i++) printf("%f ", U[i]);

    int row_tile_num = out_numrow/2, col_tile_num = out_numcol/2;
    int P = batch_size * row_tile_num * col_tile_num;
    float* V = new float[in_channels*P*16]; // C x P x 16
    for(int b=0; b<P; b++){
        for(int c=0; c<in_channels; c++){
            float d_cb[16];
            int cur_batch = b / row_tile_num / col_tile_num;
            int cur_tile_row = (b / col_tile_num) % row_tile_num;
            int cur_tile_col = b % col_tile_num;
            int in_start_row = -padding + 2*cur_tile_row;
            int in_start_col = -padding + 2*cur_tile_col;
            // prepare_tile_2x2(A, d_cb, batch_size, in_channels, in_numrow, in_numcol, in_start_row, in_start_col);

            for(int i=0; i<4; i++){
                for(int j=0; j<4; j++){
                    if(in_start_row+i >= 0 && in_start_row+i < in_numrow && in_start_col+j >= 0 && in_start_col+j < in_numcol)
                        d_cb[i*4+j] = A_b[cur_batch*in_channels*in_numrow*in_numcol + c*in_numrow*in_numcol + 
                                (in_start_row+i)*in_numcol + (in_start_col+j)]; // d_cb[i, j] = A[in_batch, c, in_start_row+i, in_start_col+j]

                    else d_cb[i*4+j]=0;
                    // printf("%d %d %d %d %f \n", cur_batch, c, in_start_row+i, in_start_col+j, d_cb[i*4+j]);
                }
            }
            float v[16];
            calc_BtdB(d_cb, v);
            for(int i=0; i<16; i++){
                V[c*(P*16) + b*16 + i] = v[i];
                // printf("%f ", v[i]);
            }
        }
    }

    // for(int i=0; i<in_channels*P*16; i++) printf("%f ", V[i]);

    float* M = new float[out_channels*P*16];
    matmul_4x4(U, V, M, out_channels, in_channels, P);

    for(int k=0; k<out_channels; k++){
        for(int b=0; b<P; b++){
            float m[16];
            for(int i=0; i<16; i++){
                m[i] = M[k*P*16 + b*16 + i];
            }

            float out[4];
            calc_AtmA(m, out);
            // k,b,4 -> B,k,w,h
            int now_batch = b / row_tile_num / col_tile_num;
            int now_tile_row = (b / col_tile_num) % row_tile_num;
            int now_tile_col = b % col_tile_num;
            for(int i=0; i<2; i++){
                for(int j=0; j<2; j++){
                    C_b[now_batch*out_channels*out_numrow*out_numcol + k*out_numrow*out_numcol + 
                        (now_tile_row*2+i)*out_numcol + now_tile_col*2 + j] = out[2*i+j]; // C_b[now_batch, k, 2*now_tile_row+i, 2*now_tile_col+j]
                }
            }
        }
    }
}

int main()
{  
    // EASY CASE
    // float kernel[18], input[8], output[4]; // kernel: 1*2*3*3, input: 1*2*2*2, output: 1*1*2*2
    // for(int i=0; i<18; i++) kernel[i] = i+1;
    // for(int i=0; i<8; i++) input[i] = i+1;
    
    // printf("start testing\n");
    // batch_trivial_conv2d_2x2_3x3(input, kernel, output,
    //     1, 2, 2, 2, 1, 1);
    
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

    // float kernel[18], input[32], output[16]; // kernel: 1*2*3*3, input: 1*2*4*4, output: 1*1*4*4
    // for(int i=0; i<18; i++) kernel[i] = i+1;
    // for(int i=0; i<32; i++) input[i] = i+1;
    
    // printf("start testing\n");
    // batch_trivial_conv2d_2x2_3x3(input, kernel, output,
    //     1, 4, 4, 2, 1, 1);
    
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
    float kernel[72], input[144], output[288]; // kernel: 4*2*3*3, input: 2*2*6*6, output: 2*4*6*6
    for(int i=0; i<72; i++) kernel[i] = i;
    for(int i=0; i<144; i++) input[i] = i;
    
    printf("start testing\n");
    batch_trivial_conv2d_2x2_3x3(input, kernel, output,
        2, 6, 6, 2, 4, 1);
    
    for(int i=0; i<2; i++){
        for(int j=0; j<4; j++){
            for(int k=0; k<6; k++){
                for(int l=0; l<6; l++){
                    float now_element = output[i*144 + j*36 + k*6 + l];
                    printf("%f ", now_element);
                }
                printf(" \n");
            }
            printf(" \n");
        }
    }
    return 0;
}