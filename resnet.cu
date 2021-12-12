//
// Created by admin on 2021/11/20.
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include "block.cu"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <tuple>
#include <map>

template<class Dtype> class Resnet18{
private:
    conv2d<Dtype> *conv1;
    Relu<Dtype> *relu;
    maxpooling2d<Dtype> *maxpool;
    GlobalAvgpooling<Dtype> *avgpool;
    Gemm<Dtype> *gemm;
    BasicBlock<Dtype> *layer1,*layer2,*layer3,*layer4,*layer5;
    Bottleneck<Dtype> *neck_layer1,*neck_layer2,*neck_layer3;
    Dtype** Weights;
    Dtype** Biases;

public:
    Resnet18(){
        conv1 = new conv2d<Dtype>(3,64,Weights[0],Biases[0],7,1,3,2);
        relu = new Relu<Dtype>{};
        maxpool = new maxpooling2d<Dtype>{3,0,1,2};
        layer1 = new BasicBlock<Dtype>{64,64,Weights[1],Biases[1],Weights[2],Biases[2]};
        layer2 = new BasicBlock<Dtype>{64,64,Weights[3],Biases[3],Weights[4],Biases[4]};
        neck_layer1 = new Bottleneck<Dtype>{64,128,Weights[5],Biases[5],Weights[6],Biases[6],Weights[7],Biases[7],2};
        layer3 = new BasicBlock<Dtype>{128,128,Weights[8],Biases[8],Weights[9],Biases[9]};
        neck_layer2 = new Bottleneck<Dtype>{128,256,Weights[10],Biases[10],Weights[11],Biases[11],Weights[12],Biases[12],2};
        layer4 = new BasicBlock<Dtype>{256,256,Weights[13],Biases[13],Weights[14],Biases[14]};
        neck_layer3 =new Bottleneck<Dtype>{256,512,Weights[15],Biases[15],Weights[16],Biases[16],Weights[17],Biases[17],2};
        layer5 = new BasicBlock<Dtype>{512,512,Weights[18],Biases[18],Weights[19],Biases[19]};
        avgpool = new GlobalAvgpooling<Dtype>{};
        gemm = new Gemm<Dtype>{512,1000,Weights[20],Biases[20]};
    }

    void forward(tensor<Dtype>* A, tensor<Dtype>*& B){
        tensor<Dtype> *tmp_out1,*tmp_out2;

        conv1->forward(A,tmp_out1);
        relu->forward(tmp_out1,tmp_out2);
        maxpool->forward(tmp_out2,tmp_out1);

        layer1->forward(tmp_out1,tmp_out2);
        layer2->forward(tmp_out2,tmp_out1);
        neck_layer1->forward(tmp_out1,tmp_out2);
        layer3->forward(tmp_out2,tmp_out1);
        neck_layer2->forward(tmp_out1,tmp_out2);
        layer4->forward(tmp_out2,tmp_out1);
        neck_layer3->forward(tmp_out1,tmp_out2);
        layer5->forward(tmp_out2,tmp_out1);

        avgpool->forward(tmp_out1,tmp_out2);
        gemm->forward(tmp_out2,tmp_out1);

        B=tmp_out1;
    }

};


int main(){


    return 0;
}


