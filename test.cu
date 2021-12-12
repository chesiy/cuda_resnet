#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include "block.cu"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <tuple>

using namespace std;

#define TILE_WIDTH 16

void matgen(float* a, int x, int y)
{
    int i, j;
    for (i = 0; i < x; i++)
    {
        for (j = 0; j < y; j++)
        {
//            a[i * y + j] = (float)rand() / RAND_MAX + (float)rand()*2 / (RAND_MAX);
            a[i*y+j] = i*y+j;
        }
    }
}

template<typename Dtype>
void print_tensor(tensor<Dtype>* Ts){
    for(int i=0;i<Ts->batch;i++){
        for(int j=0;j<Ts->channels;j++){
            for(int k=0;k<Ts->height;k++){
                for(int t=0;t<Ts->width;t++){
                    printf("%f ",Ts->data[i*(Ts->channels*Ts->width*Ts->height)+j*(Ts->width*Ts->height)+k*Ts->width+t]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main()
{
    printf("Test start!\n");

    int x = 16;
    int y = 16;
//    int z = 1024;

    float *M = (float*)malloc(sizeof(float)*x * y);

    srand(0);
    matgen(M, x, y);			//产生矩阵M

    auto *A=new tensor<float>(M,1,1,8,16);

    print_tensor<float>(A);

    tensor<float>* B;

    tuple<int,int> *kernel=new tuple<int,int>{1,1};
    tuple<int,int> *padding=new tuple<int,int>{1,1};
    tuple<int,int> *stride=new tuple<int,int>{1,1};
    tuple<int,int> *dilations=new tuple<int,int>{1,1};

    int in_channel=8, out_channel=12;
    float *W = (float*)malloc(sizeof(float) * in_channel * out_channel * get<0>(*kernel)*get<1>(*kernel));
    matgen(W, in_channel* out_channel, get<0>(*kernel) * get<1>(*kernel));
    float *Bias = (float*)malloc(sizeof(float) * out_channel);
    matgen(Bias, out_channel,1);

    in_channel = 4;
    out_channel=4;
    float *W2 = (float*)malloc(sizeof(float) * in_channel * out_channel * get<0>(*kernel)*get<1>(*kernel));
    matgen(W2, in_channel* out_channel, get<0>(*kernel) * get<1>(*kernel));
    float *Bias2 = (float*)malloc(sizeof(float) * out_channel);
    matgen(Bias2, out_channel,1);

    in_channel = 2;
    out_channel = 4;
    float *W3 = (float*)malloc(sizeof(float) * in_channel * out_channel * 1*1);
    matgen(W3, in_channel* out_channel, 1 * 1);
    float *Bias3 = (float*)malloc(sizeof(float) * out_channel);
    matgen(Bias3, out_channel,1);

    /// ====== Test MaxPooling ======
//    printf("before pooling\n");
//    maxpooling2d<float> mxp{*kernel, *padding, *stride};
//    printf("mxp ok\n");
//    mxp.forward(A,B);

    /// ====== Test Convolution ======
//    printf("before conv\n");
//    conv2d<float> conv{2,4, W, Bias,*kernel, *dialations, *padding, *stride};
//    printf("conv ok\n");
//    conv.forward(A,B);

    /// ==== Test AvgPooling ====
//    printf("before pooling\n");
//    GlobalAvgpooling<float> avgp{};
//    printf("avgp ok\n");
//    avgp.forward(A,B);

    /// ==== Test Gemm ====
//    Gemm<float> gemm{in_channel,out_channel,W,Bias};
//    gemm.forward(A,B);

    /// ==== Test BasicBlock ====
//    BasicBlock<float> basic{in_channel, out_channel, W, Bias, W2, Bias2};
//    basic.forward(A, B);

    /// ==== Test Bottleneck ====
//    Bottleneck<float> bottleneck{in_channel, out_channel, W, Bias, W2, Bias2, W3, Bias3, 2};
//    bottleneck.forward(A, B);

    /// ==== Test Resnet ====



    /// =========================
    printf("B: %d %d %d %d\n",B->height,B->width,B->channels,B->batch);
    print_tensor<float>(B);

    free(M);

    return 0;
}