//
// Created by admin on 2021/11/20.
//
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "block.cu"
#include "string.h"
#include <iostream>
#include "stdio.h"

using namespace std;

template<typename Dtype>
void print_tensor(tensor<Dtype>* Ts){
    for(int i=0;i<Ts->batch;i++){
        for(int j=0;j<Ts->channels;j++){
            for(int k=0;k<Ts->height;k++){
                for(int t=0;t<Ts->width;t++){
                    printf("%f ",Ts->data[i*(Ts->channels*Ts->width*Ts->height)+j*(Ts->width*Ts->height)+k*Ts->width+t]);
                }
                printf("\n");
//                break;
            }
            printf("\n");
//            break;
        }
        printf("\n");
    }
}

template<class Dtype> class Resnet18{
private:
    conv2d<Dtype> *conv1;
    Relu<Dtype> *relu;
    maxpooling2d<Dtype> *maxpool;
    GlobalAvgpooling<Dtype> *avgpool;
    Gemm<Dtype> *gemm;
    BasicBlock<Dtype> *layer1,*layer2,*layer3,*layer4,*layer5;
    Bottleneck<Dtype> *neck_layer1,*neck_layer2,*neck_layer3;
    map<string, Dtype*> Parameters;

public:
    Resnet18(map<string, Dtype*> param):Parameters(param){
        conv1 = new conv2d<Dtype>{3,64,Parameters["193"],Parameters["194"],7,1,3,2};
        relu = new Relu<Dtype>{};
        maxpool = new maxpooling2d<Dtype>{3,1,2};
        layer1 = new BasicBlock<Dtype>{64,64,Parameters["196"],Parameters["197"],Parameters["199"],Parameters["200"]};
        layer2 = new BasicBlock<Dtype>{64,64,Parameters["202"],Parameters["203"],Parameters["205"],Parameters["206"]};
        neck_layer1 = new Bottleneck<Dtype>{64,128,Parameters["208"],Parameters["209"],Parameters["211"],Parameters["212"],Parameters["214"],Parameters["215"],2};
        layer3 = new BasicBlock<Dtype>{128,128,Parameters["217"],Parameters["218"],Parameters["220"],Parameters["221"]};
        neck_layer2 = new Bottleneck<Dtype>{128,256,Parameters["223"],Parameters["224"],Parameters["226"],Parameters["227"],Parameters["229"],Parameters["230"],2};
        layer4 = new BasicBlock<Dtype>{256,256,Parameters["232"],Parameters["233"],Parameters["235"],Parameters["236"]};
        neck_layer3 =new Bottleneck<Dtype>{256,512,Parameters["238"],Parameters["239"],Parameters["241"],Parameters["242"],Parameters["244"],Parameters["245"],2};
        layer5 = new BasicBlock<Dtype>{512,512,Parameters["247"],Parameters["248"],Parameters["250"],Parameters["251"]};
        avgpool = new GlobalAvgpooling<Dtype>{};
        gemm = new Gemm<Dtype>{512,1000,Parameters["fc.weight"],Parameters["fc.bias"]};
    }

    void forward(tensor<Dtype>* A, tensor<Dtype>*& B){
        tensor<Dtype> *tmp_out1,*tmp_out2;
//        printf("======= forward begin =======!\n");
        conv1->forward(A,tmp_out1);
//        printf("after conv1 %d %d %d %d %f %f \n",
//               tmp_out1->batch, tmp_out1->channels,tmp_out1->height,tmp_out1->width,
//               tmp_out1->data[0], tmp_out1->data[40]);
//        print_tensor<float>(tmp_out1);
        relu->forward(tmp_out1,tmp_out2);
//        printf("after relu %d %d %d %d %f %f \n",
//               tmp_out2->batch, tmp_out2->channels,tmp_out2->height,tmp_out2->width,
//               tmp_out2->data[0], tmp_out2->data[40]);
        free(tmp_out1->data);
        free(tmp_out1);
        maxpool->forward(tmp_out2,tmp_out1);
//        printf("after maxpooling %d %d %d %d %f %f \n",
//               tmp_out1->batch, tmp_out1->channels,tmp_out1->height,tmp_out1->width,
//               tmp_out1->data[0], tmp_out1->data[40]);
//        print_tensor<float>(tmp_out1);
        free(tmp_out2->data);
        free(tmp_out2);
//        printf("======== stage 1==========\n");
        layer1->forward(tmp_out1,tmp_out2);
        free(tmp_out1->data);
        free(tmp_out1);
        layer2->forward(tmp_out2,tmp_out1);
        free(tmp_out2->data);
        free(tmp_out2);
        neck_layer1->forward(tmp_out1,tmp_out2);
        free(tmp_out1->data);
        free(tmp_out1);
        layer3->forward(tmp_out2,tmp_out1);
        free(tmp_out2->data);
        free(tmp_out2);
        neck_layer2->forward(tmp_out1,tmp_out2);
        free(tmp_out1->data);
        free(tmp_out1);
        layer4->forward(tmp_out2,tmp_out1);
        free(tmp_out2->data);
        free(tmp_out2);
        neck_layer3->forward(tmp_out1,tmp_out2);
        free(tmp_out1->data);
        free(tmp_out1);
        layer5->forward(tmp_out2,tmp_out1);
//        print_tensor<float>(tmp_out1);
        free(tmp_out2->data);
        free(tmp_out2);
//        printf("========= stage 2 ===========\n");
//        printf("before avg %f %f %d %d %d %d\n",tmp_out1->data[0], tmp_out1->data[2],
//               tmp_out1->batch, tmp_out1->channels,tmp_out1->height,tmp_out1->width);
        avgpool->forward(tmp_out1,tmp_out2);
//        printf("after avg: %f %f %d %d %d %d\n",tmp_out2->data[0], tmp_out2->data[2],
//               tmp_out2->batch, tmp_out2->channels,tmp_out2->height,tmp_out2->width);
//        print_tensor<float>(tmp_out2);
        free(tmp_out1->data);
        free(tmp_out1);
        gemm->forward(tmp_out2,tmp_out1);
//        printf("after gemm: %f %f %d %d %d %d\n",tmp_out1->data[0], tmp_out1->data[132],
//               tmp_out1->batch, tmp_out1->channels,tmp_out1->height,tmp_out1->width);
        B=tmp_out1;
    }

};


