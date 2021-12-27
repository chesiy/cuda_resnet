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

using namespace std;

class conv2d{
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dialations;
    int padding;
    int strides;
    float* Weight;
    float* Bias;

public:
    conv2d(int in_c, int out_c, float* weight, float* bias, const int kernel_sz, const int dialations, const int padding, const int strides):
            in_channels(in_c),out_channels(out_c),kernel_size(kernel_sz),dialations(dialations),padding(padding),strides(strides){

        cudaMalloc((void**)&Weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float));
        cudaMalloc((void**)&Bias, 1*1*out_channels* sizeof(float));

        cudaMemcpy((void*)Weight, (void*)weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Bias, (void*)bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

    }
    //input->tensor_A; output->tensor_B
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){

        height_B = (height_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        width_B = (width_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        channel_B = out_channels;

        cudaMalloc((void**)&B, batch * width_B * height_B * out_channels * sizeof(float));

        int nthreads = batch * width_B * height_B * out_channels;

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        ConvolutionForward<<<blockNum, threadsPerBlock>>>(A, B, Weight, Bias, nthreads,batch, height_A, width_A, in_channels ,height_B, width_B, out_channels,
                                                          kernel_size,kernel_size,strides,strides,padding,padding);
    }
};


class conv_im2col {
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dialations;
    int padding;
    int strides;
    float* Weight;
    float* Bias;
    cudaStream_t* stream;

public:
    conv_im2col(int in_c, int out_c, float* weight, float* bias, cudaStream_t* stream, const int kernel_sz, const int dialations, const int padding, const int strides):
            in_channels(in_c),out_channels(out_c),stream(stream),kernel_size(kernel_sz),dialations(dialations),padding(padding),strides(strides){

        cudaMalloc((void**)&Weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float));
        cudaMalloc((void**)&Bias, 1*1*out_channels* sizeof(float));

        cudaMemcpy((void*)Weight, (void*)weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Bias, (void*)bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

    }
    //input->tensor_A; output->tensor_B
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){

        height_B = (height_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        width_B = (width_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        channel_B = out_channels;

        float *d_inp_col;
        cudaMalloc((void**)&d_inp_col, sizeof(float) * batch*(in_channels*kernel_size*kernel_size)*(height_B*width_B));
        cudaMalloc((void**)&B, batch * width_B * height_B * out_channels * sizeof(float));

        dim3 blockNum(batch, height_B*in_channels, kernel_size*kernel_size);
        dim3 threadsPerBlock(width_B);
        if(stream == NULL){
            im2col::im2col_CHW<<<blockNum, threadsPerBlock>>>(A,d_inp_col, kernel_size, in_channels,
                    height_A, width_A, height_B, width_B, strides, padding, batch);
        }else{
            im2col::im2col_CHW<<<blockNum, threadsPerBlock, 0, *stream>>>(A,d_inp_col, kernel_size, in_channels,
                    height_A, width_A, height_B, width_B, strides, padding, batch);
        }

        const int mm_tilewidth = 8;
        dim3 blockNum2(batch, (out_channels+mm_tilewidth-1)/mm_tilewidth, (height_B*width_B+mm_tilewidth-1)/mm_tilewidth);
        dim3 threadsPerBlock2(mm_tilewidth, mm_tilewidth);
        if(stream == NULL){
            im2col::matmul_alloc_bias<<<blockNum2,threadsPerBlock2>>>(Weight, d_inp_col, Bias, B, batch, out_channels,
                    height_A, width_A,
                    out_channels,(in_channels*kernel_size*kernel_size), (height_B*width_B));
        }else{
            im2col::matmul_alloc_bias<<<blockNum2,threadsPerBlock2, 2*mm_tilewidth*mm_tilewidth, *stream>>>(Weight, d_inp_col, Bias, B, batch, out_channels,
                    height_A, width_A,
                    out_channels,(in_channels*kernel_size*kernel_size), (height_B*width_B));
        }
    }
};


class conv_wino_4x4_3x3 {
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dialations;
    int padding;
    int strides;
    float *Weight;
    float *Bias;
    float *d_U;
    float *d_V;
    float *d_UV;

public:
    conv_wino_4x4_3x3(int in_c, int out_c, float *weight, float *bias, const int kernel_sz, const int dialations,
                      const int padding, const int strides) :
            in_channels(in_c), out_channels(out_c), kernel_size(kernel_sz),
            dialations(dialations), padding(padding), strides(strides) {

        cudaMalloc((void**)&Bias, 1*1*out_channels* sizeof(float));
        cudaMemcpy((void*)Bias, (void*)bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

        float *U = (float *) malloc(sizeof(float) * out_channels * in_channels * 36); // out_channel(4)*in_channel(2)*36
        winograd4::calc_U(weight, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

        cudaMalloc((void **) &d_U, sizeof(float) * out_channels * in_channels * 36);
        cudaMemcpy((void*)d_U, (void*)U, sizeof(float) * out_channels * in_channels * 36, cudaMemcpyHostToDevice);

        printf("wino conv 4x4 weight malloc done. \n");

    }

    //input->tensor_A; output->tensor_B
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B) {

        int tile_numrow = ceil(height_A*1.0/4);
        int tile_numcol = ceil(width_A*1.0/4);
        int P = batch * tile_numrow * tile_numcol;
        int tile_num = tile_numrow * tile_numcol;
        const int mm_tilewidth = 8;
//        printf("P %d tile_num %d\n",P,tile_num);
        // =================================================计算输出大小
        height_B = (height_A + 2 * padding - dialations * (kernel_size - 1) - 1) / strides + 1;
        width_B = (width_A + 2 * padding - dialations * (kernel_size - 1) - 1) / strides + 1;
        channel_B = out_channels;

        cudaMalloc((void **) &B, batch * width_B * height_B * out_channels * sizeof(float));
        cudaMalloc((void **) &d_V, sizeof(float) * in_channels * P * 36);
        cudaMalloc((void **) &d_UV, sizeof(float) * out_channels * P * 36);

        // =================================================执行
        winograd4::calc_V<<<dim3(batch, tile_num, in_channels), dim3(6, 6)>>>(A, d_V, P, batch, in_channels, height_A, width_A, tile_numrow, tile_numcol);
        winograd4::calc_UV<<<dim3(ceil(out_channels*1.0 / mm_tilewidth),ceil(P*1.0 / mm_tilewidth), 36), dim3(mm_tilewidth, mm_tilewidth)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
        winograd4::calc_AtmA_bias<<<dim3(out_channels, batch, tile_num), dim3(6, 6)>>>(d_UV, B, Bias, out_channels, P,
                height_B, width_B, tile_num, tile_numrow,tile_numcol);

    }
};


class conv_wino_2x2_3x3{
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dialations;
    int padding;
    int strides;
    float* Weight;
    float* Bias;
    float *d_U;
    float *d_V;
    float *d_UV;

public:
    conv_wino_2x2_3x3(int in_c, int out_c, float* weight, float* bias, const int kernel_sz, const int dialations, const int padding, const int strides):
            in_channels(in_c),out_channels(out_c),kernel_size(kernel_sz),dialations(dialations),padding(padding),strides(strides){
        cudaMalloc((void**)&Bias, 1*1*out_channels* sizeof(float));
        cudaMemcpy((void*)Bias, (void*)bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

        float* U = (float*) malloc(sizeof(float)*out_channels*in_channels*16);
        winograd2::calc_U(weight, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

        cudaMalloc((void **) &d_U, sizeof(float) * out_channels * in_channels * 16);
        cudaMemcpy((void*)d_U, (void*)U, sizeof(float) * out_channels * in_channels * 16, cudaMemcpyHostToDevice);

        printf("wino conv 2x2 weight malloc done. \n");

    }
    //input->tensor_A; output->tensor_B
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){

        int tile_numrow = ceil(height_A*1.0/2);
        int tile_numcol = ceil(width_A*1.0/2);
        int P = batch * tile_numrow * tile_numcol;
        int tile_num = tile_numrow * tile_numcol;

        const int mm_tilewidth = 8;
//        printf("P %d tile_num %d\n",P,tile_num);
        // =================================================计算输出大小
        height_B = (height_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        width_B = (width_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        channel_B = out_channels;

        cudaMalloc((void**)&B, batch * width_B * height_B * out_channels * sizeof(float));
        cudaMalloc((void **) &d_V, sizeof(float) * in_channels * P * 16);
        cudaMalloc((void **) &d_UV, sizeof(float) * out_channels * P * 16);

        // =================================================执行
        winograd2::calc_V<<<dim3(batch, tile_num, in_channels), dim3(4, 4)>>>(A, d_V, P, batch, in_channels, height_A, width_A, tile_numrow, tile_numcol);
        winograd2::calc_UV<<<dim3(ceil(out_channels*1.0/mm_tilewidth), ceil(P*1.0/mm_tilewidth), 16), dim3(mm_tilewidth, mm_tilewidth)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
        winograd2::calc_AtmA_bias<<<dim3(out_channels, batch, tile_num), dim3(2, 2)>>>(d_UV, B, Bias, out_channels, P, height_B, width_B, tile_num,tile_numrow, tile_numcol);

    }
};


class maxpooling2d {
private:
    int kernel_size;
    int padding;
    int strides;

public:
    maxpooling2d(int kernel_sz, int padding, int strides):
            kernel_size(kernel_sz), padding(padding),strides(strides){}

    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){

//        float* input = (float*) malloc(sizeof(float) * height_A * width_A * channel_A * batch);
//        cudaMemcpy((void *)input, (void *) A, batch * width_A * height_A * channel_A * sizeof(float),
//                   cudaMemcpyDeviceToHost);
//        printf("check maxpooling input: %f %f\n", input[0], input[20]);
        // =================================================计算输出大小
        height_B = (height_A-kernel_size+2*padding)/strides+1;
        width_B = (width_A-kernel_size+2*padding)/strides+1;
        channel_B = channel_A;

        cudaMalloc((void**)&B, batch * width_B * height_B * channel_A * sizeof(float));

        // =================================================执行
        int nthreads = batch * width_B * height_B * channel_B;

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        MaxPoolForward <<<blockNum, threadsPerBlock>>>(A,B, nthreads, channel_A, height_A, width_A, height_B, width_B,
                                                       kernel_size,kernel_size,strides,strides,padding,padding);

    }

};

class GlobalAvgpooling{
public:
    GlobalAvgpooling()= default;
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){

//        float* input = (float*) malloc(sizeof(float) * height_A * width_A * channel_A * batch);
//        cudaMemcpy((void *)input, (void *) A, batch * width_A * height_A * channel_A * sizeof(float),
//                   cudaMemcpyDeviceToHost);
//        printf("check avgpooling input: %f %f\n", input[0], input[20]);

        // =================================================计算输出大小
        height_B = 1;
        width_B = 1;
        channel_B = channel_A;

        cudaMalloc((void**)&B, batch * width_B * height_B * channel_B * sizeof(float));

        // =================================================执行
        int nthreads = batch * width_B * height_B * channel_B;

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        AvgPoolForward<<<blockNum, threadsPerBlock>>>(A,B, nthreads,channel_A,height_A,width_A,height_B,width_B,
                                                      height_A, width_A,1,1,0,0);

//        printf("Avgpooling done! %d %d %d %d %f\n",tensor_B->batch,tensor_B->channels,tensor_B->height,tensor_B->width,tensor_B->data[0]);

    }
};


class Relu{
public:
    Relu()= default;

    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){

        height_B = height_A;
        width_B = width_A;
        channel_B = channel_A;
        cudaMalloc((void**)&B, batch * width_B * height_B * channel_B * sizeof(float));

        // =================================================执行
        int nthread = width_B * height_B * batch * channel_B;

        int num=nthread/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        relu <<<blockNum, threadsPerBlock>>>(A,B,nthread);
    }
};


class Add_Relu{
public:
    Add_Relu()= default;
    // A+B=C
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float* B, int height_B, int width_B, int channel_B,
                 float*& C, int &height_C, int &width_C, int &channel_C) {

//        float* input = (float*) malloc(sizeof(float) * height_A * width_A * channel_A * batch);
//        cudaMemcpy((void *)input, (void *) A, batch * width_A * height_A * channel_A * sizeof(float),
//                   cudaMemcpyDeviceToHost);
//        printf("check add relu input: %f %f\n", input[0], input[20]);

        height_C = height_A;
        width_C = width_A;
        channel_C = channel_A;

        cudaMalloc((void **) &C, batch * width_C * height_C * channel_C * sizeof(float));

        int nthread = width_C * height_C * batch * channel_C;

        int num=nthread/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        add_relu<<<blockNum, threadsPerBlock>>>(A, B, C,nthread);
    }
};


class Gemm{
private:
    int in_dim;
    int out_dim;
    float* Weight; // out_dim x in_dim
    float* Bias; // out_dim
public:
    Gemm(int indim, int outdim, float* weight, float* bias):in_dim(indim),out_dim(outdim){
        cudaMalloc((void**)&Weight, out_dim * in_dim * sizeof(float));
        cudaMalloc((void**)&Bias, out_dim* sizeof(float));

        cudaMemcpy((void*)Weight, (void*)weight, out_dim * in_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)Bias, (void*)bias, out_dim * sizeof(float), cudaMemcpyHostToDevice);

    }
    // A x Weight + Bias = B
    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){

//        float* input = (float*) malloc(sizeof(float) * height_A * width_A * channel_A * batch);
//        cudaMemcpy((void *)input, (void *) A, batch * width_A * height_A * channel_A * sizeof(float),
//                   cudaMemcpyDeviceToHost);
//        printf("check gemm input: %f %f\n", input[0], input[20]);

        height_B = 1;
        width_B = 1;
        channel_B = out_dim;

        cudaMalloc((void**)&B, batch * width_B * height_B * channel_B * sizeof(float));

        int nthreads = batch * out_dim;

        dim3 blockNum(batch*out_dim/400+1,1);
        dim3 threadsPerBlock(20, 20);

        simple_matmul<<<blockNum, threadsPerBlock>>>(A, B, Weight, Bias, nthreads, batch, in_dim, out_dim);
//        printf("block:%f %f %d %d\n", ceil(batch*1.0/4), ceil(out_dim*1.0/2), out_dim, batch);
//        dim3 blockNum(ceil(batch*1.0/4),ceil(out_dim*1.0/4));
//        dim3 threadsPerBlock(4, 4);
//
//        matmul<<<blockNum, threadsPerBlock>>>(A, B, Weight, Bias, nthreads, batch, in_dim, out_dim);
        //      printf("gemm done!: %f %f %d %d %d %d\n",tensor_B->data[0], tensor_B->data[132],
        //             tensor_B->batch, tensor_B->channels,tensor_B->height,tensor_B->width);
    }
};


class BasicBlock{
private:
    float* Weight1;
    float* Bias1;
    float* Weight2;
    float* Bias2;
    conv_im2col *conv1;
    conv_im2col *conv2;
    conv_wino_2x2_3x3 *conv1_2x2;
    conv_wino_2x2_3x3 *conv2_2x2;
    conv_wino_4x4_3x3 *conv1_4x4;
    conv_wino_4x4_3x3 *conv2_4x4;
    Relu *relu;
    Add_Relu *add_relu;
    int conv_type;

public:
    ~BasicBlock(){};

    BasicBlock(int _inplanes, int _planes, float* weight1, float* bias1, float* weight2, float* bias2, int conv_type):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),conv_type(conv_type)
    {
        if (conv_type == 1){
            conv1 = new conv_im2col{_inplanes, _planes, Weight1,Bias1, NULL, 3, 1, 1, 1};//3*3卷积，stride=1
            conv2 = new conv_im2col{_planes, _planes, Weight2, Bias2, NULL, 3, 1, 1, 1};//3*3卷积，stride=1
        }else if (conv_type == 2){
            conv1_2x2 = new conv_wino_2x2_3x3{_inplanes, _planes, Weight1,Bias1, 3, 1, 1, 1};
            conv2_2x2 = new conv_wino_2x2_3x3{_planes, _planes, Weight2, Bias2,3, 1, 1, 1};
        }else if (conv_type == 4){
            conv1_4x4 = new conv_wino_4x4_3x3{_inplanes, _planes, Weight1,Bias1, 3, 1, 1, 1};
            conv2_4x4 = new conv_wino_4x4_3x3{_planes, _planes, Weight2, Bias2,3, 1, 1, 1};
        }

        relu = new Relu{};
        add_relu = new Add_Relu{};
    };

    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){
        float* residual = A;
        float *output, *output2;
        int height, width, channel;
        int height2, width2, channel2;

        if (conv_type == 1){
            conv1->forward(A, height_A, width_A, channel_A, batch,
                           output, height, width, channel);
        }else if (conv_type == 2){
            conv1_2x2->forward(A, height_A, width_A, channel_A, batch,
                               output, height, width, channel);
        }else if (conv_type == 4){
            conv1_4x4->forward(A, height_A, width_A, channel_A, batch,
                               output, height, width, channel);
        }
        relu->forward(output, height, width, channel, batch,
                      output2, height2, width2, channel2);

        cudaFree(output);

        if (conv_type == 1){
            conv2->forward(output2, height2, width2, channel2, batch,
                           output,height, width, channel);
        }else if (conv_type == 2){
            conv2_2x2->forward(output2, height2, width2, channel2, batch,
                               output,height, width, channel);
        }else if (conv_type == 4){
            conv2_4x4->forward(output2, height2, width2, channel2, batch,
                               output,height, width, channel);
        }
        cudaFree(output2);

        add_relu->forward(output,height, width, channel,batch,
                          residual,height_A,width_A, channel_A,
                          B, height_B, width_B, channel_B);
        cudaFree(output);
    };
};


class Bottleneck{
private:
    float *Weight1,*Bias1;
    float *Weight2,*Bias2;
    float *Weight3,*Bias3;
    conv_im2col *conv1,*conv2,*conv3;
    conv_wino_2x2_3x3 *conv1_2x2;
    conv_wino_2x2_3x3 *conv2_2x2;
    conv_wino_4x4_3x3 *conv1_4x4;
    conv_wino_4x4_3x3 *conv2_4x4;
    Relu *relu;
    Add_Relu *add_relu;
    int conv_type;
    cudaStream_t* stream;

public:
    ~Bottleneck(){};

    Bottleneck(int _inplanes, int _planes, float* weight1, float* bias1, float* weight2, float* bias2, float* weight3, float* bias3,
               cudaStream_t* stream, int _stride, int conv_type):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),Weight3(weight3),Bias3(bias3),conv_type(conv_type),stream(stream)
    {
        conv1 = new conv_im2col{_inplanes,_planes,weight1,bias1,NULL,3,1,1,_stride};//3*3卷积 stride=_strinde ic=_inplanes oc=width
        if (conv_type == 1){
            conv2 = new conv_im2col{_planes,_planes,weight2,bias2,NULL, 3, 1, 1, 1};//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        }else if (conv_type == 2){
            conv2_2x2 = new conv_wino_2x2_3x3{_planes,_planes,weight2,bias2, 3, 1, 1, 1};//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        }else if (conv_type == 4){
            conv2_4x4 = new conv_wino_4x4_3x3{_planes,_planes,weight2,bias2, 3, 1, 1, 1};//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        }

        conv3 = new conv_im2col{_inplanes,_planes,weight3,bias3,stream, 1, 1, 0, _stride};//1*1 ic=width,oc=_planes*expansion
        relu = new Relu;
        add_relu = new Add_Relu;
    };

    void forward(float* A, int height_A, int width_A, int channel_A, int batch,
                 float*& B, int& height_B, int &width_B, int &channel_B){
        float* identity  = A;
        float *output, *output2;
        int height, width, channel;
        int height2, width2, channel2;

        conv1->forward(A, height_A, width_A, channel_A, batch,
                           output, height, width, channel);

        relu->forward(output, height, width, channel, batch,
                      output2, height2, width2, channel2);
        cudaFree(output);

        if (conv_type == 1){
            conv2->forward(output2, height2, width2, channel2, batch,
                           output,height, width, channel);
        }else if (conv_type == 2){
            conv2_2x2->forward(output2, height2, width2, channel2, batch,
                               output,height, width, channel);
        }else if (conv_type == 4){
            conv2_4x4->forward(output2, height2, width2, channel2, batch,
                               output,height, width, channel);
        }

        cudaFree(output2);

        conv3->forward(identity, height_A, width_A, channel_A, batch,
                       output2, height2, width2, channel2);

        cudaStreamSynchronize(*stream);

        add_relu->forward(output2,height2, width2, channel2, batch,
                          output, height, width, channel,
                          B, height_B, width_B, channel_B);

        cudaFree(output2);
        cudaFree(output);

    };
};