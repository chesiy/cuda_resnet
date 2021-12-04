//
// Created by admin on 2021/12/4.
//
#include <iostream>
#include <cuda.h>
#include <string.h>
#include <tuple>
#include "kernels.cu"

using namespace std;

template<class Dtype> struct tensor{
    Dtype* data;
    int width,height,channels,batch;
    //tensor shape: (batch, channels, width, height)
};

template<class Dtype> class conv2d{
private:
    int in_channels;
    int out_channels;
    tuple<int,int> kernel_size;
    tuple<int,int> dialations;
    tuple<int,int> padding;
    tuple<int,int> strides;
    Dtype* Weight;
    Dtype* Bias;

public:
    conv2d(int in_c, int out_c, Dtype* weight, Dtype* bias, const tuple<int,int>&kernel_sz, const tuple<int,int> &dialations={1,1}, const tuple<int,int>&padding={0,0}, const tuple<int,int>&strides={1,1}):
            in_channels(in_c),out_channels(out_c),Weight(weight),Bias(bias),kernel_size(kernel_sz),dialations(dialations),padding(padding),strides(strides){}
    //input->tensor_A; output->tensor_B
    void forward(const tensor<Dtype>* tensor_A, const tensor<Dtype>* tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width;
        const int batch = tensor_A->batch;
        Dtype *A=tensor_A->data;

        // =================================================计算输出大小
        int height_B = (height_A+2*get<0>(padding)-get<0>(dialations)*(get<0>(kernel_size)-1)-1)/get<0>(strides) + 1;
        int width_B = (width_A+2*get<1>(padding)-get<1>(dialations)*(get<1>(kernel_size)-1)-1)/get<1>(strides) + 1;

        tensor_B->height = height_B;
        tensor_B->width = width_B;
        tensor_B->channels = out_channels;

        const Dtype* d_A, d_B, d_K, d_bias;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * in_channels * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * out_channels * sizeof(float));
        cudaMalloc((void**)&d_K, get<0>(kernel_size)* get<1>(kernel_size) * in_channels * out_channels * sizeof(float));
        cudaMalloc((void**)&d_bias, 1*1*out_channels* sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * in_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_K, (void*)Weight, get<0>(kernel_size)* get<1>(kernel_size) * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_bias, (void*)Bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        // =================================================执行
        int nthreads = batch * width_B * height_B * out_channels;
        ConvolutionForward(d_A, d_B, Weight,Bias, nthreads,in_channels, height_A, width_A, height_B, width_B,
                           get<0>(kernel_size),get<1>(kernel_size),get<0>(strides),get<1>(strides),get<0>(padding),get<1>(padding));

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * sizeof(Dtype), cudaMemcpyDeviceToHost);

    }
};


template<class Dtype> class maxpooling2d {
private:
    tuple<int, int> kernel_size;
    tuple<int, int> padding;
    tuple<int, int> strides;

public:
    maxpooling2d(const tuple<int,int>&kernel_sz, const tuple<int,int>&padding={0,0}, const tuple<int,int>&strides={1,1}):
            kernel_size(kernel_sz), padding(padding),strides(strides){}

    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>* tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width,channels_A=tensor_A->channels;
        Dtype *A=tensor_A->data;
        const int batch = tensor_A->batch;

        // =================================================计算输出大小
        int height_B = (height_A-get<0>(kernel_size)+2*get<0>(padding))/get<0>(strides)+1;
        int width_B = (width_A-get<1>(kernel_size)+2*get<1>(padding))/get<1>(strides)+1;
        tensor_B->height=height_B;
        tensor_B->width=width_B;
        tensor_B->channels = channels_A;

        const Dtype* d_A, d_B;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * channels_A * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * channels_A * sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * channels_A * sizeof(float), cudaMemcpyHostToDevice);
        // =================================================执行
        int nthreads = batch * width_B * height_B * channels_A;
        MaxPoolForward(d_A,d_B,0,NULL, nthreads, channels_A, height_A, width_A, height_B, width_B,
                       get<0>(kernel_size), get<1>(kernel_size),get<0>(strides),get<1>(strides),get<0>(padding),get<1>(padding));

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * channels_A *sizeof(float), cudaMemcpyDeviceToHost);

    }

};

template<class Dtype> class GlobalAvgpooling{
public:
    GlobalAvgpooling()= default;
    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>* tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width,channels_A=tensor_A->channels;
        Dtype *A=tensor_A->data;
        const int batch = tensor_A->batch;

        // =================================================计算输出大小
        int height_B = 1;
        int width_B = 1;
        tensor_B->height=1;
        tensor_B->width=1;
        tensor_B->channels = channels_A;

        const Dtype* d_A, d_B;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * channels_A * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * channels_A * sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * channels_A * sizeof(float), cudaMemcpyHostToDevice);
        // =================================================执行
        int nthreads = batch * width_B * height_B * channels_A;
        AvgPoolForward(d_A,d_B,0,NULL,nthreads,channels_A,height_A,width_A,height_B,width_B,
                       height_A, width_A,0,0,1,1);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * channels_A *sizeof(float), cudaMemcpyDeviceToHost);

    }
};


template<class Dtype> class Relu{
public:
    Relu()= default;

    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>* tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width, channels_A=tensor_A->channels;
        Dtype *A=tensor_A->data;
        const int batch = tensor_A->batch;

        const Dtype* d_A, d_B, d_K;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * sizeof(Dtype));
        cudaMalloc((void**)&d_B, batch * width_A * height_A * sizeof(Dtype));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * sizeof(Dtype), cudaMemcpyHostToDevice);

        // =================================================执行
        relu(d_A,d_B,width_A,height_A);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_A * height_A * sizeof(Dtype), cudaMemcpyDeviceToHost);
        tensor_B->height=height_A;
        tensor_B->width=width_A;
        tensor_B->channels=channels_A;
    }
};


template<class Dtype> class Add{
    Add()= default;
    // A+B=C
    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>* tensor_B, tensor<Dtype>* tensor_C) {
        const int height=tensor_A->height, width=tensor_A->width, channels=tensor_A->channels;
        Dtype *A=tensor_A->data;
        Dtype *B=tensor_B->data;
        const int batch = tensor_A->batch;

        const Dtype *d_A, d_B, d_C;
        cudaMalloc((void **) &d_A, batch * width * height * channels * sizeof(float));
        cudaMalloc((void **) &d_B, batch * width * height * channels * sizeof(float));
        cudaMalloc((void **) &d_C, batch * width * height * channels * sizeof(float));

        cudaMemcpy((void *) d_A, (void *) A, batch * width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *) d_B, (void *) B, batch * width * height * channels * sizeof(float), cudaMemcpyHostToDevice);

        add(d_A, d_B, d_C,width,height);

        cudaMemcpy((void *) tensor_C->data, (void *) d_C, batch * width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);
        tensor_C->height=height;
        tensor_C->width=width;
        tensor_C->channels=channels;

    }
};

template<class Dtype> class BasicBlock{
private:
    int expansion = 1;

    Dtype* Weight1;
    Dtype* Bias1;
    Dtype* Weight2;
    Dtype* Bias2;
    conv2d<Dtype> conv1,conv2;
    Relu<Dtype> relu;
    Add<Dtype> add;

public:
    ~BasicBlock(){};

    BasicBlock(int _inplanes, int _planes, Dtype* weight1, Dtype* bias1, Dtype* weight2, Dtype* bias2, int _stride, int _groups=1,int _base_width=64,int _dilation=1):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2)
    {
        conv1(_inplanes, _planes, Weight1,Bias1,{3,3},{1,1},{1,1},{_stride,_stride});//3*3卷积，stride=_stride
        relu();
        conv2(_planes, _planes, Weight2, Bias2,{3,3},{1,1},{1,1},{1,1});//3*3卷积，stride=1
        add();
    };

    tensor<Dtype>* forward(tensor<Dtype>* A){
        tensor<Dtype>* residual  = A;
        tensor<Dtype>* output, output2;//没有返回值为了节约所以交替使用了hhh 现在是第一个是输入，第二个是输出，需要按照具体类的实现改

        conv1.forward(A,output);
        relu.forward(output,output2);
        conv2.forward(output2,output);

        add.forward(output,residual,output2); //output2=output+residual
        relu.forward(output2,output);

        return output;
    };
};

template<class Dtype> class Bottleneck{
private:
    int expansion = 4;

    Dtype* Weight1,Bias1;
    Dtype* Weight2,Bias2;
    Dtype* Weight3,Bias3;
    conv2d<Dtype> conv1,conv2,conv3;
    Relu<Dtype> relu;
    Add<Dtype> add;

public:
    ~Bottleneck(){};

    Bottleneck(int _inplanes, int _planes, Dtype* weight1, Dtype* bias1, Dtype* weight2, Dtype* bias2, Dtype* weight3, Dtype* bias3,int _stride, int _groups=1,int _base_width=64,int _dilation=1):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),Weight3(weight3),Bias3(bias3)
    {
//        int width = int(_planes*(_base_width/64.0))*_groups;
        //初始化的参数应该改成现在类对应的
        conv1(_inplanes,_planes,weight1,bias1,{3,3},{1,1},{1,1},{_stride,_stride});//3*3卷积 stride=_strinde ic=_inplanes oc=width
        conv2(_planes,_planes,weight2,bias2,{3,3},{1,1},{1,1},{1,1});//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        conv3(_inplanes,_planes,weight3,bias3,{1,1},{1,1},{0,0},{_stride,_stride});//1*1 ic=width,oc=_planes*expansion
        relu();
        add();
    };

    tensor<Dtype>* forward(tensor<Dtype>* A){
        Dtype* identity  = A;
        Dtype* output,output2,output3;//没有返回值为了节约所以交替使用了hhh 现在是第一个是输入，第二个是输出，需要按照具体类的实现改

        conv1.forward(A,output);
        relu.forward(output2,output);
        conv2.forward(output,output2);

        conv3.forward(identity,output);

        add.forward(output2,output,output3); //output3=output+output2
        relu(output3,output);

        return output;
    };

};
