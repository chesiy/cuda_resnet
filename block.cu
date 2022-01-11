//
// Created by admin on 2021/12/4.
//
#include <iostream>
#include <cuda.h>
#include <string.h>
#include "kernels.cu"
#include "conv_winograd_4x4_3x3.cu"
#include "conv_winograd_gpu.cu"
#include "conv_im2col.cu"
#include "conv_im2col_transpose.cu"
#include "matmul.cu"

using namespace std;

class conv{
public:
    virtual void forward(float* A, float*& B)=0;
};

/*
TOO LAZY TO USE ENUMERATE
conv_type:
    0: im2col, pre-allocate weight
    1: im2col, allocate weight in forward
    2: winograd 4x4, pre-allocate weight,
    3: winograd 4x4 + relu, pre-allocate weight,
    4: im2col + relu, pre-allocate weight
    5: im2col + add + relu, pre-allocate weight,
    6: winograd 4x4 + add + relu, pre-allocate weight,
    7: im2col + add + relu + avg_pooling, pre-allocate weight,
    -1: im2col, pre-allocate weight, only for 7x7
*/

void check_status(cudaError_t cudaStatus, const char* info){
    if (cudaStatus != cudaSuccess) {
        printf("%s ", info);
        printf("memery operation failed %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
}


class conv_im2col: public conv {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int padding;
    int strides;
    int inp_size;
    int out_size;
    int batch;
    int colmat_height;
    int colmat_width;
    float* Weight;
    float* Bias;
    float* ColMat;
    float* Out;
    int conv_type;

    conv_im2col(int in_c, int out_c, float* weight, float* bias, int conv_type, const int kernel_sz, const int padding, const int stride, const int inp_size, const int batch):
            in_channels(in_c),out_channels(out_c), conv_type(conv_type),kernel_size(kernel_sz),padding(padding),strides(stride), inp_size(inp_size), batch(batch){

        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc((void**)&Weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float));
        check_status(cudaStatus, "conv_im2col_weight");

        cudaStatus = cudaMalloc((void**)&Bias, 1*1*out_channels* sizeof(float));
        check_status(cudaStatus, "conv_im2col_bias");

        cudaMemcpy((void*)Weight, (void*)weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Bias, (void*)bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

        colmat_height = in_c * kernel_size * kernel_size;
        out_size = (inp_size + padding*2 - kernel_size) / stride + 1;
        colmat_width = out_size*out_size;
        
        if(conv_type==-1 || conv_type == 0 || conv_type == 4 || conv_type == 5 || conv_type == 7){
            cudaStatus = cudaMalloc((void**)&ColMat, sizeof(float) * 1*colmat_height*colmat_width);
            check_status(cudaStatus, "conv_im2col_malloc1");
        }

        if(conv_type==-1 || conv_type == 0 || conv_type == 4){
            cudaStatus = cudaMalloc((void**)&Out, sizeof(float) * 1*out_c*out_size*out_size);
            check_status(cudaStatus, "conv_im2col_malloc2");
        }
    }
    
    //input->tensor_A; output->tensor_B
    void forward(float* A, float*& B){
        
        cudaError_t cudaStatus;
        if(! (conv_type==-1 || conv_type == 0 || conv_type == 4 || conv_type == 5 || conv_type == 7)){
            cudaStatus = cudaMalloc((void**)&ColMat, sizeof(float) * batch*colmat_height*colmat_width);
            check_status(cudaStatus, "conv_im2col malloc in forward");
            cudaStatus = cudaMalloc((void**)&Out, batch * out_size * out_size * out_channels * sizeof(float));
            check_status(cudaStatus, "conv_im2col malloc in forward");
        }

        // 第一个卷积
        if(conv_type==-1){
            dim3 blockNum(batch, out_size*in_channels, kernel_size);
            dim3 threadsPerBlock(out_size, kernel_size);
            im2col::im2col_CHW<<<blockNum, threadsPerBlock>>>(A, ColMat, kernel_size, in_channels,
                    inp_size, inp_size, out_size, out_size, strides, padding, batch);
        }else{ //其他卷积
            dim3 blockNum(batch, out_size ,in_channels);
            dim3 threadsPerBlock(out_size, kernel_size, kernel_size);

            im2col::im2col_CHW_2<<<blockNum, threadsPerBlock>>>(A, ColMat, kernel_size, in_channels,
                    inp_size, inp_size, out_size, out_size, strides, padding, batch);
        }

        const int mm_tilewidth = 8;
        dim3 blockNum2(batch, (out_channels+mm_tilewidth-1)/mm_tilewidth, (out_size*out_size+mm_tilewidth-1)/mm_tilewidth);
        dim3 threadsPerBlock2(mm_tilewidth, mm_tilewidth);

        switch(conv_type){
            case 4:
                im2col::matmul_alloc_bias_relu<<<blockNum2,threadsPerBlock2>>>(Weight, ColMat, Bias, Out, batch, out_channels,
                    out_size, out_size,
                    out_channels,(in_channels*kernel_size*kernel_size), (out_size*out_size));
                B = Out;
                break;
            case 5: // in-place operation
                im2col::matmul_alloc_bias_add_relu<<<blockNum2,threadsPerBlock2>>>(Weight, ColMat, Bias, B, batch, out_channels,
                    out_size, out_size, out_channels,(in_channels*kernel_size*kernel_size), (out_size*out_size));
                break;
            case 7: // in-place operation
                im2col::matmul_alloc_bias_add_relu<<<blockNum2,threadsPerBlock2>>>(Weight, ColMat, Bias, B, batch, out_channels,
                    out_size, out_size, out_channels,(in_channels*kernel_size*kernel_size), (out_size*out_size));
                break;
            case -1: // 7x7
                // im2col::matmul_alloc_bias_relu_maxpool<<<blockNum2,threadsPerBlock2>>>(Weight, ColMat, Bias, Out, batch, out_channels,
                //     inp_size, inp_size, out_channels,(in_channels*kernel_size*kernel_size), (out_size*out_size));
                im2col::matmul_alloc_bias_relu<<<blockNum2,threadsPerBlock2>>>(Weight, ColMat, Bias, Out, batch, out_channels,
                    out_size, out_size, out_channels,(in_channels*kernel_size*kernel_size), (out_size*out_size));
                B = Out;
                break;
            default: // no post processing
                im2col::matmul_alloc_bias<<<blockNum2,threadsPerBlock2>>>(Weight, ColMat, Bias, Out, batch, out_channels,
                    out_size, out_size,
                    out_channels,(in_channels*kernel_size*kernel_size), (out_size*out_size));
                B = Out;
                break;
        }

        if(! (conv_type==-1 || conv_type == 0 || conv_type == 4 || conv_type == 5 || conv_type == 7)){
            cudaStatus = cudaFree(ColMat);
            check_status(cudaStatus, "conv_im2col free in forward");
        }
    }
};


class conv_wino_4x4_3x3: public conv {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int padding;
    int strides;
    int conv_type;
    int P;
    int P36;
    int inp_size;
    int out_size;
    int tile_numrow;
    int tile_numcol;
    int tile_num;
    int batch;
    int out_enum;
    int inp_enum;

    float *Weight;
    float *Bias;
    float *d_U;
    float *d_V;
    float *d_UV;
    float *Out;

    conv_wino_4x4_3x3(int in_c, int out_c, float* weight, float* bias, int conv_type, const int kernel_sz, const int padding, const int stride, const int inp_size, const int batch):
            in_channels(in_c), out_channels(out_c), conv_type(conv_type), kernel_size(kernel_sz), padding(padding), strides(stride), inp_size(inp_size), batch(batch) {

        cudaMalloc((void**)&Bias, 1*1*out_channels* sizeof(float));
        cudaMemcpy((void*)Bias, (void*)bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

        float *U = (float *) malloc(sizeof(float) * out_channels * in_channels * 36); // out_channel(4)*in_channel(2)*36
        winograd4::calc_U(weight, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

        cudaMalloc((void **) &d_U, sizeof(float) * out_channels * in_channels * 36);
        cudaMemcpy((void*)d_U, (void*)U, sizeof(float) * out_channels * in_channels * 36, cudaMemcpyHostToDevice);

        tile_numrow = (inp_size+3)/4;
        tile_numcol = (inp_size+3)/4;
        P = batch * tile_numrow * tile_numcol;
        P36 = P*36;
        out_size = (inp_size + padding*2 - kernel_size) / stride + 1;
        tile_num = tile_numrow * tile_numcol;
        out_enum = out_size * out_size *  out_channels;
        inp_enum = inp_size * inp_size * in_channels;
        cudaMalloc((void**)&d_V, sizeof(float) * P*in_channels*36);
        cudaMalloc((void**)&d_UV, sizeof(float) * out_c*P*36);
        cudaMalloc((void**)&Out, sizeof(float) * batch*out_c*out_size*out_size);
    }

    //input->tensor_A; output->tensor_B
    void forward(float* A, float*& B) {

        const int mm_tilewidth = 4;
        // =================================================执行
        winograd4::calc_V<<<dim3(batch, tile_num, in_channels), dim3(6, 6)>>>
                (A, d_V, P, batch, in_channels, inp_size, inp_size, tile_numrow, tile_numcol, tile_num, inp_size*inp_size, inp_enum, P36);
        
        switch(conv_type){
            case 2:
                winograd4::calc_UV_AtmA_bias<<<dim3((P+mm_tilewidth-1)/mm_tilewidth, (out_channels+mm_tilewidth-1)/mm_tilewidth), dim3(36, mm_tilewidth, mm_tilewidth)>>>
                    (d_U, d_V, Out, Bias, out_channels, in_channels, P, out_size, out_size, tile_num, tile_numrow, tile_numcol, out_size*out_size, out_enum, P36);
                B = Out;
                break;
            case 3:
                winograd4::calc_UV_AtmA_bias_relu<<<dim3((P+mm_tilewidth-1)/mm_tilewidth, (out_channels+mm_tilewidth-1)/mm_tilewidth), dim3(36, mm_tilewidth, mm_tilewidth)>>>
                    (d_U, d_V, Out, Bias, out_channels, in_channels, P, out_size, out_size, tile_num, tile_numrow, tile_numcol, out_size*out_size, out_enum, P36);
                B = Out;
                break;
            case 6:
                winograd4::calc_UV_AtmA_bias_add_relu<<<dim3((P+mm_tilewidth-1)/mm_tilewidth, (out_channels+mm_tilewidth-1)/mm_tilewidth), dim3(36, mm_tilewidth, mm_tilewidth)>>>
                    (d_U, d_V, B, Bias, out_channels, in_channels, P, out_size, out_size, tile_num, tile_numrow, tile_numcol, out_size*out_size, out_enum, P36);
                break;
        }
    }
};


class maxpooling2d {
private:
    int kernel_size;
    int padding;
    int strides;
    float* B;

public:
    maxpooling2d(int kernel_sz, int padding, int strides):
            kernel_size(kernel_sz), padding(padding),strides(strides){
        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc((void**)&B, 1*64*56*56 * sizeof(float));
        check_status(cudaStatus, "maxpooling malloc");
    }

    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& Out, int height_B, int width_B, int channel_B){

        // =================================================执行
        int nthreads = batch * width_B * height_B * channel_B;

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        MaxPoolForward <<<blockNum, threadsPerBlock>>>(A, B, nthreads, channel_A, height_A, width_A, height_B, width_B,
                                                       kernel_size,kernel_size,strides,strides,padding,padding);
        Out = B;
    }

};


class GlobalAvgpooling{
public:
    float * B;
    GlobalAvgpooling(){
        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc((void**)&B, 1*512*1*1 * sizeof(float));
        check_status(cudaStatus, "avgpooling malloc");
    }
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& Out, int height_B, int width_B, int channel_B){

        // =================================================执行
        int nthreads = batch * width_B * height_B * channel_B;

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        AvgPoolForward<<<blockNum, threadsPerBlock>>>(A,B, nthreads,channel_A,height_A,width_A,height_B,width_B,
                                                      height_A, width_A,1,1,0,0);

        Out = B;
    }
};


class Gemm{
private:
    int l1, l2, l3;
    float* Weight; // out_dim x in_dim
    float* Bias; // out_dim
    float* B;
public:
    Gemm(int l1, int l2, int l3, float* weight, float* bias):
        l1(l1), l2(l2), l3(l3){
        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc((void**)&Weight, l2 * l3 * sizeof(float));
        check_status(cudaStatus, "gemm weight malloc");
        cudaStatus = cudaMalloc((void**)&Bias, l3* sizeof(float));
        check_status(cudaStatus, "gemm bias malloc");

        cudaMemcpy((void*)Weight, (void*)weight, l2 * l3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Bias, (void*)bias, l3 * sizeof(float), cudaMemcpyHostToDevice);

        cudaStatus = cudaMalloc((void**)&B, l1 * l3 * sizeof(float));
        check_status(cudaStatus, "gemm malloc");
    }

    // A x Weight + Bias = B
    void forward(float* A, float*& Out){

        const int mm_tilewidth = 8;
        dim3 blockNum((l1+mm_tilewidth-1)/mm_tilewidth, (l3+mm_tilewidth-1)/mm_tilewidth);
        dim3 threadsPerBlock(mm_tilewidth, mm_tilewidth);
        matmul::matmul_bias<<<blockNum,threadsPerBlock>>>(A, Weight, Bias, B, l1, l2, l3);
        Out = B;
    }
};


class BasicBlock{
private:
    float* Weight1;
    float* Bias1;
    float* Weight2;
    float* Bias2;
    conv *conv1;
    conv *conv2;
    int conv_type;

public:
    ~BasicBlock(){};

    BasicBlock(int _inplanes, int _planes, float* weight1, float* bias1, float* weight2, float* bias2, int conv_type1, int conv_type2, int _inp_size):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),conv_type(conv_type)
    {
        if (conv_type1==0 || conv_type1==1 || conv_type1==4 || conv_type1==5 || conv_type1==7)
            conv1 = new conv_im2col{_inplanes, _planes, Weight1, Bias1, conv_type1, 3, 1, 1, _inp_size, 1};//3*3卷积，stride=1
        else 
            conv1 = new conv_wino_4x4_3x3{_inplanes, _planes, Weight1, Bias1, conv_type1, 3, 1, 1, _inp_size, 1};

        if (conv_type2==0 || conv_type2==1 || conv_type2==4 || conv_type2==5 || conv_type2==7)
            conv2 = new conv_im2col{_planes, _planes, Weight2, Bias2, conv_type2, 3, 1, 1, _inp_size, 1};//3*3卷积，stride=1
        else
            conv2 = new conv_wino_4x4_3x3{_planes, _planes, Weight2, Bias2, conv_type2, 3, 1, 1, _inp_size, 1};
    };

    void forward(float* A, float*& B){
        float *output, *output2;
        output = A;

        conv1->forward(A, output2);
        conv2->forward(output2, output);

        B = output;
    };
};


class Bottleneck{
private:
    float *Weight1,*Bias1;
    float *Weight2,*Bias2;
    float *Weight3,*Bias3;
    conv *conv1, *conv2, *conv3;
    int conv_type1, conv_type2, conv_type3;

public:
    ~Bottleneck(){};

    Bottleneck(int _inplanes, int _planes, float* weight1, float* bias1, float* weight2, float* bias2, float* weight3, float* bias3,int _stride, int conv_type1, int conv_type2, int conv_type3, int _inp_size):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),Weight3(weight3),Bias3(bias3),conv_type1(conv_type1), conv_type2(conv_type2), conv_type3(conv_type3)
    {
        if (conv_type1==0 || conv_type1==1 || conv_type1==4 || conv_type1==5 || conv_type1==7)
            conv1 = new conv_im2col{_inplanes, _planes, Weight1, Bias1, conv_type1, 3, 1, 2, _inp_size, 1}; //3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        else 
            conv1 = new conv_wino_4x4_3x3{_inplanes, _planes, Weight1, Bias1, conv_type1, 3, 1, 2,  _inp_size, 1};

        if (conv_type2==0 || conv_type2==1 || conv_type2==4 || conv_type2==5 || conv_type2==7)
            conv2 = new conv_im2col{_planes, _planes, Weight2, Bias2, conv_type2, 3, 1, 1, _inp_size/2, 1};//3*3卷积，stride=1
        else
            conv2 = new conv_wino_4x4_3x3{_planes, _planes, Weight2, Bias2, conv_type2, 3, 1, 1,  _inp_size/2, 1};
        
        if (conv_type3==0 || conv_type3==1 || conv_type3==4 || conv_type3==5 || conv_type3==7)
            conv3 = new conv_im2col{_inplanes, _planes, Weight3, Bias3, conv_type3, 1, 0, 2, _inp_size, 1};//1*1卷积，stride=1
        else
            conv3 = new conv_wino_4x4_3x3{_inplanes, _planes, Weight3, Bias3, conv_type3, 1, 0, 2,  _inp_size, 1};
    };

    void forward(float* A, float*& B){
        float *output1, *output2;

        conv1->forward(A, output1);
        conv2->forward(output1, output2);

        conv3->forward(A, output2);
        
        B = output2;
    };

};