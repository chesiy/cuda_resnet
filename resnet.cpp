//
// Created by admin on 2021/11/20.
//
#include <iostream>
#include "myblocks.cpp"

template<class Dtype> class Resnet18{
private:
    conv2d<Dtype> conv1,conv2,conv3;
    Relu<Dtype> relu;
    pooling2d<Dtype> maxpool;
    pooling2d<Dtype> avgpool;
    fullyconnect<Dtype> fc;
    Dtype* conv1_w,conv1_b;
    BasicBlock<Dtype> layer1,layer2,layer3,layer4,layer5,layer6,layer7,layer8;
    Dtype* w1,w2,b1,b2;

public:
    Resnet18(){
        conv1(3,64,conv1_w,conv1_b,{7,7},{1,1},{3,3},{2,2});
        relu();
        maxpool({3,3},0,{1,1},{2,2});
        layer1(64,64,w1,b1,w2,b2,1);
        layer2(64,64,w1,b1,w2,b2,1);
        layer3(64,128,w1,b1,w2,b2,2);
        layer4(128,128,w1,b1,w2,b2,1);
        layer5(128,256,w1,b1,w2,b2,2);
        layer6(256,256,w1,b1,w2,b2,1);
        layer7(256,512,w1,b1,w2,b2,2);
        layer8(512,1512,w1,b1,w2,b2,1);
    }

    void forward(Dtype* A, Dtype* B){
        Dtype *tmp_out1,*tmp_out2;
        int out_height1,out_width1,out_channels1;
        int out_height2,out_width2,out_channels2;
        conv1.forward(A,tmp_out1,224,224,&out_height1,&out_width1,&out_channels1);
        relu.forward(tmp_out1,tmp_out2,out_height1,out_width1,out_channels1);
        maxpool.forward(tmp_out2,tmp_out1,out_height1,out_width1,out_channels1,&out_height2,&out_width2,&out_channels2);

        layer1.forward(tmp_out1)
    }

};
