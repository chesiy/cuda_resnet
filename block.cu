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
    tensor(Dtype* d, int w, int h, int c, int batch):data(d),width(w),height(h),channels(c),batch(batch){}
    tensor(const tensor<Dtype> &d){
        data = d.data;
        width = d.width;
        height = d.height;
        batch = d.batch;
    }
    tensor(){}
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
    conv2d(int in_c, int out_c, Dtype* weight, Dtype* bias, const tuple<int,int>&kernel_sz, const tuple<int,int> &dialations, const tuple<int,int>&padding, const tuple<int,int>&strides):
            in_channels(in_c),out_channels(out_c),Weight(weight),Bias(bias),kernel_size(kernel_sz),dialations(dialations),padding(padding),strides(strides){}
    //input->tensor_A; output->tensor_B
    void forward(const tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width;
        const int batch = tensor_A->batch;
        Dtype *A=tensor_A->data;
//        printf("A: %d %d %d %d %f\n",height_A,width_A,in_channels,batch, A[0]);

        // =================================================计算输出大小
        int height_B = (height_A+2*get<0>(padding)-get<0>(dialations)*(get<0>(kernel_size)-1)-1)/get<0>(strides) + 1;
        int width_B = (width_A+2*get<1>(padding)-get<1>(dialations)*(get<1>(kernel_size)-1)-1)/get<1>(strides) + 1;

        Dtype* B = (float*)malloc(sizeof(float)*height_B*width_B*out_channels*batch);
        tensor_B=new tensor<float>(B,width_B,height_B,out_channels,batch);

//        printf("B: %d %d %d %f\n",tensor_B->height,tensor_B->width,tensor_B->channels,tensor_B->data[0]);

        Dtype* d_A;
        Dtype* d_B;
        Dtype* d_K;
        Dtype* d_bias;
//        printf("start cuda malloc\n");
        cudaMalloc((void**)&d_A, batch * width_A * height_A * in_channels * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * out_channels * sizeof(float));
        cudaMalloc((void**)&d_K, get<0>(kernel_size)* get<1>(kernel_size) * in_channels * out_channels * sizeof(float));
        cudaMalloc((void**)&d_bias, 1*1*out_channels* sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * in_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_K, (void*)Weight, get<0>(kernel_size)* get<1>(kernel_size) * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_bias, (void*)Bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);
//        printf("cuda cpy ok\n");
        // =================================================执行
        int nthreads = batch * width_B * height_B * out_channels;

        dim3 blockNum(batch, out_channels);
        dim3 threadsPerBlock(width_B, height_B);

        ConvolutionForward<<<blockNum, threadsPerBlock>>>(d_A, d_B, d_K, d_bias, nthreads,batch, height_A, width_A, in_channels ,height_B, width_B, out_channels,
                           get<0>(kernel_size),get<1>(kernel_size),get<0>(strides),get<1>(strides),get<0>(padding),get<1>(padding));

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * out_channels * sizeof(Dtype), cudaMemcpyDeviceToHost);
    }
};


template<class Dtype> class maxpooling2d {
private:
    tuple<int, int> kernel_size;
    tuple<int, int> padding;
    tuple<int, int> strides;

public:
    maxpooling2d(tuple<int,int>& kernel_sz, tuple<int,int>& padding, tuple<int,int>&strides):
            kernel_size(kernel_sz), padding(padding),strides(strides){}

    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width,channels_A=tensor_A->channels;
        Dtype *A=tensor_A->data;
        const int batch = tensor_A->batch;
        printf("A: %d %d %d %d \n",height_A,width_A,channels_A,batch);

        // =================================================计算输出大小
        int height_B = (height_A-get<0>(kernel_size)+2*get<0>(padding))/get<0>(strides)+1;
        int width_B = (width_A-get<1>(kernel_size)+2*get<1>(padding))/get<1>(strides)+1;

        Dtype* B = (Dtype*)malloc(sizeof(Dtype)*height_B*width_B*channels_A*batch);
        tensor_B=new tensor<float>(B,width_B,height_B,channels_A,batch);

        printf("B: %d %d %d\n",tensor_B->height,tensor_B->width,tensor_B->channels);
        Dtype* d_A;
        Dtype* d_B;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * channels_A * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * channels_A * sizeof(float));

        printf("cuda malloc ok\n");
        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * channels_A * sizeof(float), cudaMemcpyHostToDevice);
        printf("cuda cpy ok\n");

        // =================================================执行
        int nthreads = batch * width_B * height_B * channels_A;

        dim3 blockNum(batch, channels_A);
        dim3 threadsPerBlock(width_B, height_B);

        MaxPoolForward<Dtype> <<<blockNum, threadsPerBlock>>>(d_A,d_B, nthreads, channels_A, height_A, width_A, height_B, width_B,
                       get<0>(kernel_size), get<1>(kernel_size),get<0>(strides),get<1>(strides),get<0>(padding),get<1>(padding));

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * channels_A *sizeof(float), cudaMemcpyDeviceToHost);

//        printf("B::: %f\n",tensor_B->data[0]);

    }

};

template<class Dtype> class GlobalAvgpooling{
public:
    GlobalAvgpooling()= default;
    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width,channels_A=tensor_A->channels;
        Dtype *A=tensor_A->data;
        const int batch = tensor_A->batch;

        // =================================================计算输出大小
        int height_B = 1;
        int width_B = 1;

        Dtype* B = (Dtype*)malloc(sizeof(Dtype)*height_B*width_B*channels_A*batch);
        tensor_B=new tensor<float>(B,width_B,height_B,channels_A,batch);

        Dtype* d_A;
        Dtype* d_B;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * channels_A * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * channels_A * sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * channels_A * sizeof(float), cudaMemcpyHostToDevice);
        // =================================================执行
        int nthreads = batch * width_B * height_B * channels_A;

        dim3 blockNum(batch, channels_A);
        dim3 threadsPerBlock(width_B, height_B);

        AvgPoolForward<<<blockNum, threadsPerBlock>>>(d_A,d_B, nthreads,channels_A,height_A,width_A,height_B,width_B,
                       height_A, width_A,0,0,1,1);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * channels_A *sizeof(float), cudaMemcpyDeviceToHost);

    }
};


template<class Dtype> class Relu{
public:
    Relu()= default;

    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width, channels_A=tensor_A->channels;
        Dtype *A=tensor_A->data;
        const int batch = tensor_A->batch;

        Dtype* B = (Dtype*)malloc(sizeof(Dtype)*height_A*width_A*channels_A*batch);
        tensor_B=new tensor<float>(B,width_A,height_A,channels_A,batch);

        Dtype* d_A;
        Dtype* d_B;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * channels_A * sizeof(Dtype));
        cudaMalloc((void**)&d_B, batch * width_A * height_A * channels_A * sizeof(Dtype));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * channels_A * sizeof(Dtype), cudaMemcpyHostToDevice);

        // =================================================执行
        int nthread = width_A * height_A * batch * channels_A;

        dim3 blockNum(batch, channels_A);
        dim3 threadsPerBlock(width_A, height_A);
        relu<Dtype> <<<blockNum, threadsPerBlock>>>(d_A,d_B,nthread);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_A * height_A * channels_A * sizeof(Dtype), cudaMemcpyDeviceToHost);

    }
};


template<class Dtype> class Add{
public:
    Add()= default;
    // A+B=C
    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>* tensor_B, tensor<Dtype>*& tensor_C) {
        const int height=tensor_A->height, width=tensor_A->width, channels=tensor_A->channels;
        Dtype *A=tensor_A->data;
        Dtype *B=tensor_B->data;
        const int batch = tensor_A->batch;

        Dtype* C = (Dtype*)malloc(sizeof(Dtype)*height*width*channels*batch);
        tensor_C = new tensor<float>(C,width,height,channels,batch);

        Dtype *d_A;
        Dtype *d_B;
        Dtype *d_C;
        cudaMalloc((void **) &d_A, batch * width * height * channels * sizeof(float));
        cudaMalloc((void **) &d_B, batch * width * height * channels * sizeof(float));
        cudaMalloc((void **) &d_C, batch * width * height * channels * sizeof(float));

        cudaMemcpy((void *) d_A, (void *) A, batch * width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *) d_B, (void *) B, batch * width * height * channels * sizeof(float), cudaMemcpyHostToDevice);

        int nthread = width * height * batch * channels;
        dim3 blockNum(batch, channels);
        dim3 threadsPerBlock(width, height);

        add<<<blockNum, threadsPerBlock>>>(d_A, d_B, d_C,nthread);

        cudaMemcpy((void *) tensor_C->data, (void *) d_C, batch * width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    }
};

template<class Dtype> class BasicBlock{
private:
    Dtype* Weight1;
    Dtype* Bias1;
    Dtype* Weight2;
    Dtype* Bias2;
    conv2d<Dtype> *conv1,*conv2;
    Relu<Dtype> *relu;
    Add<Dtype> *add;

public:
    ~BasicBlock(){};

    BasicBlock(int _inplanes, int _planes, Dtype* weight1, Dtype* bias1, Dtype* weight2, Dtype* bias2):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2)
    {
        tuple<int,int> *kernel=new tuple<int,int>{3, 3};
        tuple<int,int> *one=new tuple<int,int>{1, 1};

        conv1 = new conv2d<Dtype>{_inplanes, _planes, Weight1,Bias1,*kernel, *one, *one ,*one};//3*3卷积，stride=1
        relu = new Relu<Dtype>{};
        conv2 = new conv2d<Dtype>{_planes, _planes, Weight2, Bias2,*kernel, *one,*one,*one};//3*3卷积，stride=1
        add = new Add<Dtype>{};
    };

    void forward(tensor<Dtype>* A, tensor<Dtype>*& B){
        tensor<Dtype>* residual = new tensor<Dtype>(*A);
        tensor<Dtype> *output, *output2;

        conv1->forward(A,output);
//        printf("conv ok %d %d %d %d %f \n", output->batch,output->channels,output->height,output->width, output->data[131]);
        relu->forward(output,output2);
//        printf("relu ok output %d %d %d %d %f \n",output2->batch,output2->channels,output2->height,output2->width, output2->data[131]);
        free(output->data);
        free(output);
        conv2->forward(output2,output);
//        printf("conv2 ok %d %d %d %d %f \n", output->batch,output->channels,output->height,output->width, output->data[131]);
        free(output2->data);
        free(output2);
//        printf("before add %f %f %f \n",output->data[131],residual->data[131],A->data[131]);
        add->forward(output,residual,output2); //output2=output+residual
//        printf("add ok %d %d %d %d %f \n",output2->batch,output2->channels,output2->height,output2->width, output2->data[131]);
        free(output->data);
        free(output);
        relu->forward(output2,output);
//        printf("relu ok\n");
        free(output2->data);
        free(output2);

        B = output;
//        printf("B: %d %d %d %d\n",B->batch,B->channels,B->height,B->width);
    };
};

template<class Dtype> class Bottleneck{
private:
    int expansion = 4;

    Dtype *Weight1,*Bias1;
    Dtype *Weight2,*Bias2;
    Dtype *Weight3,*Bias3;
    conv2d<Dtype> *conv1,*conv2,*conv3;
    Relu<Dtype> *relu;
    Add<Dtype> *add;

public:
    ~Bottleneck(){};

    Bottleneck(int _inplanes, int _planes, Dtype* weight1, Dtype* bias1, Dtype* weight2, Dtype* bias2, Dtype* weight3, Dtype* bias3,int _stride):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),Weight3(weight3),Bias3(bias3)
    {
        tuple<int,int> *kernel=new tuple<int,int>{3,3};
        tuple<int,int> *one=new tuple<int,int>{1,1};
        tuple<int,int> *zero=new tuple<int,int>{0,0};
        tuple<int,int> *stride=new tuple<int,int>{_stride, _stride};

        conv1 = new conv2d<Dtype>{_inplanes,_planes,weight1,bias1,*kernel,*one,*one,*stride};//3*3卷积 stride=_strinde ic=_inplanes oc=width
        conv2 = new conv2d<Dtype>{_planes,_planes,weight2,bias2,*kernel, *one, *one, *one};//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        conv3 = new conv2d<Dtype>{_inplanes,_planes,weight3,bias3,*one, *one, *zero,*stride};//1*1 ic=width,oc=_planes*expansion
        relu = new Relu<Dtype>;
        add = new Add<Dtype>;
    };

    void forward(tensor<Dtype>* A, tensor<Dtype>*& B){
        tensor<Dtype>* identity  = new tensor<Dtype>(*A);
        tensor<Dtype> *output, *output2, *output3;
        printf("start!\n");
        conv1->forward(A,output);
        printf("conv ok %d %d %d %d %f \n", output->batch,output->channels,output->height,output->width, output->data[131]);
        relu->forward(output,output2);
        printf("relu ok output %d %d %d %d %f \n",output2->batch,output2->channels,output2->height,output2->width, output2->data[131]);
        free(output->data);
        free(output);
        conv2->forward(output2,output);
        printf("conv2 ok %d %d %d %d %f \n", output->batch,output->channels,output->height,output->width, output->data[131]);

        conv3->forward(identity,output2);

        add->forward(output2,output,output3); //output3=output+output2
        free(output2->data);
        free(output2);
        free(output->data);
        free(output);
        relu->forward(output3,output);
        free(output3->data);
        free(output3);

        B = output;

    };

};
