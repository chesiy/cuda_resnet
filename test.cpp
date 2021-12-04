#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include "block.cpp"
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
            a[i * x + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX*RAND_MAX);
        }
    }
}

double Try_GPU()
{
    float *M, *N, *Pg;
    int x = 64;	//1024×1024矩阵乘法
    int y = 64;
    int z = 1024;
    M = (float*)malloc(sizeof(float)* x * y);
    N = (float*)malloc(sizeof(float)* y * z);
    Pg = (float*)malloc(sizeof(float)* x * z); //保存GPU计算结果

    srand(0);

    matgen(M, x, y);			//产生矩阵M
    matgen(N, y, z);			//产生矩阵N

    double timeStart, timeEnd;	//定义时间，求时间差用
    timeStart = clock();
    auto *A=new tensor<float>(M,16,16,4,4);
    tensor<float>* B;
    //MatrixMultiplication_CUDA(M, N, Pg, x, y, z, gamma);			//GPU上计算
    tuple<int,int> *kernel=new tuple<int,int>{2,2};
    tuple<int,int> *padding=new tuple<int,int>{0,0};
    tuple<int,int> *stride=new tuple<int,int>{1,1};
    maxpooling2d<float> mxp{*kernel, *padding, *stride};

    mxp.forward(A,B);

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