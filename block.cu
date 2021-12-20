//
// Created by admin on 2021/12/4.
//
#include <iostream>
#include <cuda.h>
#include <string.h>
#include "kernels.cu"
#include "tensor.cu"
#include "conv_winograd_4x4_3x3.cu"
#include "conv_winograd_gpu.cu"

using namespace std;

template<class Dtype> class conv2d{
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dialations;
    int padding;
    int strides;
    Dtype* Weight;
    Dtype* Bias;

public:
    conv2d(int in_c, int out_c, Dtype* weight, Dtype* bias, const int kernel_sz, const int dialations, const int padding, const int strides):
            in_channels(in_c),out_channels(out_c),Weight(weight),Bias(bias),kernel_size(kernel_sz),dialations(dialations),padding(padding),strides(strides){}
    //input->tensor_A; output->tensor_B
    void forward(const tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width;
        const int batch = tensor_A->batch;
        Dtype *A=tensor_A->data;
        // =================================================计算输出大小
        int height_B = (height_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        int width_B = (width_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;

        Dtype* B = (float*)malloc(sizeof(float)*height_B*width_B*out_channels*batch);
        tensor_B=new tensor<float>(B,width_B,height_B,out_channels,batch);

        Dtype* d_A;
        Dtype* d_B;
        Dtype* d_K;
        Dtype* d_bias;

        cudaMalloc((void**)&d_A, batch * width_A * height_A * in_channels * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * out_channels * sizeof(float));
        cudaMalloc((void**)&d_K, kernel_size*kernel_size * in_channels * out_channels * sizeof(float));
        cudaMalloc((void**)&d_bias, 1*1*out_channels* sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * in_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_K, (void*)Weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_bias, (void*)Bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

        // =================================================执行
        int nthreads = batch * width_B * height_B * out_channels;

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        ConvolutionForward<<<blockNum, threadsPerBlock>>>(d_A, d_B, d_K, d_bias, nthreads,batch, height_A, width_A, in_channels ,height_B, width_B, out_channels,
                           kernel_size,kernel_size,strides,strides,padding,padding);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * out_channels * sizeof(Dtype), cudaMemcpyDeviceToHost);

    }
};


template<class Dtype> class conv_wino_4x4_3x3 {
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dialations;
    int padding;
    int strides;
    Dtype *Weight;
    Dtype *Bias;

public:
    conv_wino_4x4_3x3(int in_c, int out_c, Dtype *weight, Dtype *bias, const int kernel_sz, const int dialations,
                      const int padding, const int strides) :
            in_channels(in_c), out_channels(out_c), Weight(weight), Bias(bias), kernel_size(kernel_sz),
            dialations(dialations), padding(padding), strides(strides) {}

    //input->tensor_A; output->tensor_B
    void forward(const tensor<Dtype> *tensor_A, tensor<Dtype> *&tensor_B) {
        const int height_A = tensor_A->height, width_A = tensor_A->width;
        const int batch = tensor_A->batch;
        Dtype *A = tensor_A->data;
        int P = batch * ceil(height_A/4) * ceil(width_A/4);
        int tile_num = ceil(height_A/4) * ceil(width_A/4) ;
        // =================================================计算输出大小
        int height_B = (height_A + 2 * padding - dialations * (kernel_size - 1) - 1) / strides + 1;
        int width_B = (width_A + 2 * padding - dialations * (kernel_size - 1) - 1) / strides + 1;

        Dtype *B = (float *) malloc(sizeof(float) * height_B * width_B * out_channels * batch);
        tensor_B = new tensor<float>(B, width_B, height_B, out_channels, batch);

        Dtype *d_A;
        Dtype *d_B;
//        Dtype *d_K;
        Dtype *d_bias;

        float *U = (float *) malloc(sizeof(float) * out_channels * in_channels * 36); // out_channel(4)*in_channel(2)*36
        winograd4::calc_U(Weight, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

        Dtype *d_V, *d_U, *d_UV;

        cudaMalloc((void **) &d_A, batch * width_A * height_A * in_channels * sizeof(float));
        cudaMalloc((void **) &d_B, batch * width_B * height_B * out_channels * sizeof(float));
        //        cudaMalloc((void**)&d_K, kernel_size*kernel_size * in_channels * out_channels * sizeof(float));
        cudaMalloc((void **) &d_bias, 1 * 1 * out_channels * sizeof(float));

        cudaMemcpy((void *) d_A, (void *) A, batch * width_A * height_A * in_channels * sizeof(float),
                   cudaMemcpyHostToDevice);
        //        cudaMemcpy((void*)d_K, (void*)Weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void *) d_bias, (void *) Bias, 1 * 1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void **) &d_V, sizeof(float) * in_channels * P * 36);
        cudaMalloc((void **) &d_U, sizeof(float) * out_channels * in_channels * 36);
        cudaMalloc((void **) &d_UV, sizeof(float) * out_channels * P * 36);
        cudaMemcpy(d_U, U, sizeof(float) * out_channels * in_channels * 36, cudaMemcpyHostToDevice);

        // =================================================执行
        winograd4::calc_V<<<dim3(batch, tile_num, in_channels), dim3(6, 6)>>>(d_A, d_V, P, batch, in_channels, height_A, width_A);
        winograd4::calc_UV<<<dim3(out_channels / 2, P / 2, 36), dim3(2, 2)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
        winograd4::calc_AtmA_bias<<<dim3(out_channels, batch, tile_num), dim3(6, 6)>>>(d_UV, d_B, d_bias, out_channels, P,
                                                                            height_B, width_B, tile_num);

        cudaMemcpy((void *) tensor_B->data, (void *) d_B, batch * width_B * height_B * out_channels * sizeof(Dtype),
                   cudaMemcpyDeviceToHost);
    }
};


template<class Dtype> class conv_wino_2x2_3x3{
private:
    int in_channels;
    int out_channels;
    int kernel_size;
    int dialations;
    int padding;
    int strides;
    Dtype* Weight;
    Dtype* Bias;

public:
    conv_wino_2x2_3x3(int in_c, int out_c, Dtype* weight, Dtype* bias, const int kernel_sz, const int dialations, const int padding, const int strides):
            in_channels(in_c),out_channels(out_c),Weight(weight),Bias(bias),kernel_size(kernel_sz),dialations(dialations),padding(padding),strides(strides){}
    //input->tensor_A; output->tensor_B
    void forward(const tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width;
        const int batch = tensor_A->batch;
        Dtype *A=tensor_A->data;
        int P = batch * ceil(height_A/2) * ceil(width_A/2);
        int tile_num = ceil(height_A/2) * ceil(width_A/2) ;
        // =================================================计算输出大小
        int height_B = (height_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;
        int width_B = (width_A+2*padding-dialations*(kernel_size-1)-1)/strides + 1;

        Dtype* B = (float*)malloc(sizeof(float)*height_B*width_B*out_channels*batch);
        tensor_B=new tensor<float>(B,width_B,height_B,out_channels,batch);

        Dtype* d_A;
        Dtype* d_B;
//        Dtype* d_K;
        Dtype* d_bias;

        float* U = (float*) malloc(sizeof(float)*out_channels*in_channels*16); // out_channel(4)*in_channel(2)*36
        winograd2::calc_U(Weight, U, in_channels, out_channels); // CPU function, as it can be calculated beforehand

        cudaMalloc((void**)&d_A, batch * width_A * height_A * in_channels * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * out_channels * sizeof(float));
        //        cudaMalloc((void**)&d_K, kernel_size*kernel_size * in_channels * out_channels * sizeof(float));
        cudaMalloc((void**)&d_bias, 1*1*out_channels* sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * in_channels * sizeof(float), cudaMemcpyHostToDevice);
        //        cudaMemcpy((void*)d_K, (void*)Weight, kernel_size*kernel_size * in_channels * out_channels * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_bias, (void*)Bias, 1*1 * out_channels * sizeof(float), cudaMemcpyHostToDevice);

        Dtype *d_V, *d_U, *d_UV;

        cudaMalloc((void**)&d_V, sizeof(float) * in_channels*P*16);
        cudaMalloc((void**)&d_U, sizeof(float) * out_channels*in_channels*16);
        cudaMalloc((void**)&d_UV, sizeof(float) * out_channels*P*16);
        cudaMemcpy(d_U, U, sizeof(float) * out_channels*in_channels*16, cudaMemcpyHostToDevice);

        // =================================================执行
        winograd2::calc_V<<<dim3(batch, tile_num, in_channels), dim3(4, 4)>>>(d_A, d_V, P, batch, in_channels, height_A, width_A);
        winograd2::calc_UV<<<dim3(out_channels/2, P/2, 16), dim3(2, 2)>>>(d_U, d_V, d_UV, out_channels, in_channels, P);
        winograd2::calc_AtmA_bias<<<dim3(out_channels, batch, tile_num), dim3(2, 2)>>>(d_UV, d_B, d_bias, out_channels, P, height_B, width_B, tile_num);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * out_channels * sizeof(Dtype), cudaMemcpyDeviceToHost);
    }
};


template<class Dtype> class maxpooling2d {
private:
    int kernel_size;
    int padding;
    int strides;

public:
    maxpooling2d(int kernel_sz, int padding, int strides):
            kernel_size(kernel_sz), padding(padding),strides(strides){}

    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int height_A=tensor_A->height, width_A=tensor_A->width,channels_A=tensor_A->channels;
        Dtype *A=tensor_A->data;
        const int batch = tensor_A->batch;
      //  printf("A: %d %d %d %d \n",height_A,width_A,channels_A,batch);

        // =================================================计算输出大小
        int height_B = (height_A-kernel_size+2*padding)/strides+1;
        int width_B = (width_A-kernel_size+2*padding)/strides+1;

        Dtype* B = (Dtype*)malloc(sizeof(Dtype)*height_B*width_B*channels_A*batch);
        tensor_B=new tensor<float>(B,width_B,height_B,channels_A,batch);

//        printf("B: %d %d %d\n",tensor_B->height,tensor_B->width,tensor_B->channels);
        Dtype* d_A;
        Dtype* d_B;
        cudaMalloc((void**)&d_A, batch * width_A * height_A * channels_A * sizeof(float));
        cudaMalloc((void**)&d_B, batch * width_B * height_B * channels_A * sizeof(float));

//        printf("cuda malloc ok\n");
        cudaMemcpy((void*)d_A, (void*)A, batch * width_A * height_A * channels_A * sizeof(float), cudaMemcpyHostToDevice);
//        printf("cuda cpy ok\n");

        // =================================================执行
        int nthreads = batch * width_B * height_B * channels_A;

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        MaxPoolForward<Dtype> <<<blockNum, threadsPerBlock>>>(d_A,d_B, nthreads, channels_A, height_A, width_A, height_B, width_B,
                       kernel_size,kernel_size,strides,strides,padding,padding);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * channels_A *sizeof(float), cudaMemcpyDeviceToHost);

 //       printf("Maxpooling done! %d %d %d %d %f\n",tensor_B->batch,tensor_B->channels,tensor_B->height,tensor_B->width,tensor_B->data[0]);

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

        int num=nthreads/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        AvgPoolForward<<<blockNum, threadsPerBlock>>>(d_A,d_B, nthreads,channels_A,height_A,width_A,height_B,width_B,
                       height_A, width_A,1,1,0,0);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B, batch * width_B * height_B * channels_A *sizeof(float), cudaMemcpyDeviceToHost);

//        printf("Avgpooling done! %d %d %d %d %f\n",tensor_B->batch,tensor_B->channels,tensor_B->height,tensor_B->width,tensor_B->data[0]);

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

        int num=nthread/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

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

        int num=nthread/400+1;
        dim3 blockNum(num, 1);
        dim3 threadsPerBlock(20, 20);

        add<<<blockNum, threadsPerBlock>>>(d_A, d_B, d_C,nthread);

        cudaMemcpy((void *) tensor_C->data, (void *) d_C, batch * width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);
    }
};


template<class Dtype> class Gemm{
private:
    int in_dim;
    int out_dim;
    Dtype* Weight; // out_dim x in_dim
    Dtype* Bias; // out_dim
public:
    Gemm(int indim, int outdim, Dtype* weight, Dtype* bias):in_dim(indim),out_dim(outdim),Weight(weight),Bias(bias){}
    // A x Weight + Bias = B
    void forward(tensor<Dtype>* tensor_A, tensor<Dtype>*& tensor_B){
        const int batch = tensor_A->batch;
        Dtype *A=tensor_A->data;

        Dtype* B = (float*)malloc(sizeof(float)*out_dim*batch);
        tensor_B=new tensor<float>(B,1,1,out_dim,batch);

        Dtype* d_A;
        Dtype* d_B;
        Dtype* d_W;
        Dtype* d_bias;
//        printf("start cuda malloc\n");
        cudaMalloc((void**)&d_A, batch * in_dim * sizeof(float));
        cudaMalloc((void**)&d_B, batch * out_dim * sizeof(float));
        cudaMalloc((void**)&d_W, out_dim * in_dim * sizeof(float));
        cudaMalloc((void**)&d_bias, out_dim * sizeof(float));

        cudaMemcpy((void*)d_A, (void*)A, batch * in_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_W, (void*)Weight, out_dim * in_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)d_bias, (void*)Bias, out_dim * sizeof(float), cudaMemcpyHostToDevice);

        int nthreads = batch * out_dim;

        dim3 blockNum(batch*out_dim/400+1,1);
        dim3 threadsPerBlock(20, 20);

        simple_matmul<<<blockNum, threadsPerBlock>>>(d_A, d_B, d_W, d_bias, nthreads, batch, in_dim, out_dim);

        cudaMemcpy((void*)tensor_B->data, (void*)d_B,  batch * out_dim * sizeof(Dtype), cudaMemcpyDeviceToHost);

  //      printf("gemm done!: %f %f %d %d %d %d\n",tensor_B->data[0], tensor_B->data[132],
  //             tensor_B->batch, tensor_B->channels,tensor_B->height,tensor_B->width);
    }
};

template<class Dtype> class BasicBlock{
private:
    Dtype* Weight1;
    Dtype* Bias1;
    Dtype* Weight2;
    Dtype* Bias2;
    conv2d<Dtype> *conv1;
    conv2d<Dtype> *conv2;
    conv_wino_2x2_3x3<Dtype> *conv1_2x2;
    conv_wino_2x2_3x3<Dtype> *conv2_2x2;
    conv_wino_4x4_3x3<Dtype> *conv1_4x4;
    conv_wino_4x4_3x3<Dtype> *conv2_4x4;
    Relu<Dtype> *relu;
    Add<Dtype> *add;
    int conv_type;

public:
    ~BasicBlock(){};

    BasicBlock(int _inplanes, int _planes, Dtype* weight1, Dtype* bias1, Dtype* weight2, Dtype* bias2, int conv_type):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),conv_type(conv_type)
    {
        if (conv_type == 1){
            conv1 = new conv2d<Dtype>{_inplanes, _planes, Weight1,Bias1, 3, 1, 1, 1};//3*3卷积，stride=1
            conv2 = new conv2d<Dtype>{_planes, _planes, Weight2, Bias2,3, 1, 1, 1};//3*3卷积，stride=1
        }else if (conv_type == 2){
            conv1_2x2 = new conv_wino_2x2_3x3<Dtype>{_inplanes, _planes, Weight1,Bias1, 3, 1, 1, 1};
            conv2_2x2 = new conv_wino_2x2_3x3<Dtype>{_planes, _planes, Weight2, Bias2,3, 1, 1, 1};
        }else if (conv_type == 4){
            conv1_4x4 = new conv_wino_4x4_3x3<Dtype>{_inplanes, _planes, Weight1,Bias1, 3, 1, 1, 1};
            conv2_4x4 = new conv_wino_4x4_3x3<Dtype>{_planes, _planes, Weight2, Bias2,3, 1, 1, 1};
        }

        relu = new Relu<Dtype>{};
        add = new Add<Dtype>{};
    };

    void forward(tensor<Dtype>* A, tensor<Dtype>*& B){
        tensor<Dtype>* residual = new tensor<Dtype>(*A);
        tensor<Dtype> *output, *output2;

        if (conv_type == 1){
            conv1->forward(A,output);
        }else if (conv_type == 2){
            conv1_2x2->forward(A,output);
        }else if (conv_type == 4){
            conv1_4x4->forward(A,output);
        }
//        printf("conv ok %d %d %d %d %f \n", output->batch,output->channels,output->height,output->width, output->data[131]);
        relu->forward(output,output2);
//        printf("relu ok output %d %d %d %d %f \n",output2->batch,output2->channels,output2->height,output2->width, output2->data[131]);
        free(output->data);
        free(output);

        if (conv_type == 1){
            conv2->forward(output2,output);
        }else if (conv_type == 2){
            conv2_2x2->forward(output2,output);
        }else if (conv_type == 4){
            conv2_4x4->forward(output2,output);
        }

//        conv2->forward(output2,output);
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
 //       printf("Basic block ok: %d %d %d %d\n",B->batch,B->channels,B->height,B->width);
    };
};


template<class Dtype> class Bottleneck{
private:
    Dtype *Weight1,*Bias1;
    Dtype *Weight2,*Bias2;
    Dtype *Weight3,*Bias3;
    conv2d<Dtype> *conv1,*conv2,*conv3;
    conv_wino_2x2_3x3<Dtype> *conv1_2x2;
    conv_wino_2x2_3x3<Dtype> *conv2_2x2;
    conv_wino_4x4_3x3<Dtype> *conv1_4x4;
    conv_wino_4x4_3x3<Dtype> *conv2_4x4;
    Relu<Dtype> *relu;
    Add<Dtype> *add;
    int conv_type;

public:
    ~Bottleneck(){};

    Bottleneck(int _inplanes, int _planes, Dtype* weight1, Dtype* bias1, Dtype* weight2, Dtype* bias2, Dtype* weight3, Dtype* bias3,int _stride, int conv_type):
            Weight1(weight1),Bias1(bias1),Weight2(weight2),Bias2(bias2),Weight3(weight3),Bias3(bias3),conv_type(conv_type)
    {
        if (conv_type == 1){
            conv1 = new conv2d<Dtype>{_inplanes,_planes,weight1,bias1,3,1,1,_stride};//3*3卷积 stride=_strinde ic=_inplanes oc=width
            conv2 = new conv2d<Dtype>{_planes,_planes,weight2,bias2, 3, 1, 1, 1};//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        }else if (conv_type == 2){
            conv1_2x2 = new conv_wino_2x2_3x3<Dtype>{_inplanes,_planes,weight1,bias1,3, 1, 1, _stride};//3*3卷积 stride=_strinde ic=_inplanes oc=width
            conv2_2x2 = new conv_wino_2x2_3x3<Dtype>{_planes,_planes,weight2,bias2, 3, 1, 1, 1};//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        }else if (conv_type == 4){
            conv1_4x4 = new conv_wino_4x4_3x3<Dtype>{_inplanes,_planes,weight1,bias1,3,1,1,_stride};//3*3卷积 stride=_strinde ic=_inplanes oc=width
            conv2_4x4 = new conv_wino_4x4_3x3<Dtype>{_planes,_planes,weight2,bias2, 3, 1, 1, 1};//3*3卷积，stride=1,ic\oc=width,groups=_groups,dilation=_dilation
        }

        conv3 = new conv2d<Dtype>{_inplanes,_planes,weight3,bias3, 1, 1, 0, _stride};//1*1 ic=width,oc=_planes*expansion
        relu = new Relu<Dtype>;
        add = new Add<Dtype>;
    };

    void forward(tensor<Dtype>* A, tensor<Dtype>*& B){
        tensor<Dtype>* identity  = new tensor<Dtype>(*A);
        tensor<Dtype> *output, *output2, *output3;
//        printf("start bottleneck!\n");

        if (conv_type == 1){
            conv1->forward(A,output);
        }else if (conv_type == 2){
            conv1_2x2->forward(A,output);
        }else if (conv_type == 4){
            conv1_4x4->forward(A,output);
        }
//        conv1->forward(A,output);
//        printf("conv ok %d %d %d %d %f \n", output->batch,output->channels,output->height,output->width, output->data[131]);
        relu->forward(output,output2);
//        printf("relu ok output %d %d %d %d %f \n",output2->batch,output2->channels,output2->height,output2->width, output2->data[131]);
        free(output->data);
        free(output);

        if (conv_type == 1){
            conv2->forward(output2,output);
        }else if (conv_type == 2){
            conv2_2x2->forward(output2,output);
        }else if (conv_type == 4){
            conv2_4x4->forward(output2,output);
        }
//        conv2->forward(output2,output);

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

   //     printf("Bottle neck ok %d %d %d %d %f \n", B->batch,B->channels,B->height,B->width, B->data[131]);

    };

};