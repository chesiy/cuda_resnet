//
// Created by admin on 2021/11/20.
//
#include <iostream>
#include "myblocks.cpp"

template<class Dtype> class Resnet18{
private:
    conv2d<Dtype> *conv1;
    Relu<Dtype> relu;
    maxpooling2d<Dtype> maxpool;
    GlobalAvgpooling<Dtype> avgpool;
    fullyconnect<Dtype> fc;
    Dtype* conv1_w,conv1_b;
    BasicBlock<Dtype> layer1,layer2,layer3,layer4,layer5;
    Bottleneck<Dtype> neck_layer1,neck_layer2,neck_layer3;
    Dtype* w1,w2,w3,b1,b2,b3;

public:
    Resnet18(){
        conv1(3,64,conv1_w,conv1_b,{7,7},{1,1},{3,3},{2,2});
        relu();
        maxpool({3,3},0,{1,1},{2,2});
        layer1(64,64,w1,b1,w2,b2,1);
        layer2(64,64,w1,b1,w2,b2,1);
        neck_layer1(64,128,w1,b1,w2,b2,w3,b3,2);
        layer3(128,128,w1,b1,w2,b2,1);
        neck_layer2(128,256,w1,b1,w2,b2,w3,b3,2);
        layer4(256,256,w1,b1,w2,b2,1);
        neck_layer3(256,512,w1,b1,w2,b2,w3,b3,2);
        layer5(512,512,w1,b1,w2,b2,1);
        avgpool();
        fc(512,1000,w1,b1);
    }

    void forward(tensor<Dtype>* A, tensor<Dtype>* B){
        tensor<Dtype> *tmp_out1,*tmp_out2;
        conv1.forward(A,tmp_out1);
        relu.forward(tmp_out1,tmp_out2);
        maxpool.forward(tmp_out2,tmp_out1);

        tmp_out2 = layer1.forward(tmp_out1);
        tmp_out1=layer2.forward(tmp_out2);
        tmp_out2=neck_layer1.forward(tmp_out1);
        tmp_out1=layer3.forward(tmp_out2);
        tmp_out2=neck_layer2.forward(tmp_out1);
        tmp_out1=layer4.forward(tmp_out2);
        tmp_out2=neck_layer3.forward(tmp_out1);
        tmp_out1=layer5.forward(tmp_out2);

        avgpool.forward(tmp_out1,tmp_out2);
        //TODO flatten
        fc.forward(tmp_out2,tmp_out1);

        B=tmp_out1;
    }

};
