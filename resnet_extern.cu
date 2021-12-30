#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "block.cu"
#include "string.h"
#include "stdio.h"
#include <map>

namespace resnet{
    conv_im2col*conv1;
    maxpooling2d *maxpool;
    GlobalAvgpooling *avgpool;
    Gemm *gemm;
    Relu* relu;
    BasicBlock *layer1,*layer2,*layer3,*layer4,*layer5;
    Bottleneck *neck_layer1,*neck_layer2,*neck_layer3;

    void resnet_init(map<string, float*> Parameters){
        conv1 = new conv_im2col{3,64,Parameters["193"],Parameters["194"], true,true, 7,1,3,2};
        relu = new Relu{};
        maxpool = new maxpooling2d{3,1,2};
        layer1 = new BasicBlock{64,64,Parameters["196"],Parameters["197"],Parameters["199"],Parameters["200"],1};
        layer2 = new BasicBlock{64,64,Parameters["202"],Parameters["203"],Parameters["205"],Parameters["206"],1};
        neck_layer1 = new Bottleneck{64,128,Parameters["208"],Parameters["209"],Parameters["211"],Parameters["212"],Parameters["214"],Parameters["215"],2,1};
        layer3 = new BasicBlock{128,128,Parameters["217"],Parameters["218"],Parameters["220"],Parameters["221"],1};
        neck_layer2 = new Bottleneck{128,256,Parameters["223"],Parameters["224"],Parameters["226"],Parameters["227"],Parameters["229"],Parameters["230"],2,1};
        layer4 = new BasicBlock{256,256,Parameters["232"],Parameters["233"],Parameters["235"],Parameters["236"],1};
        neck_layer3 =new Bottleneck{256,512,Parameters["238"],Parameters["239"],Parameters["241"],Parameters["242"],Parameters["244"],Parameters["245"],2,1};
        layer5 = new BasicBlock{512,512,Parameters["247"],Parameters["248"],Parameters["250"],Parameters["251"],1};
        avgpool = new GlobalAvgpooling{};
        gemm = new Gemm{512,1000,Parameters["fc.weight"],Parameters["fc.bias"]};
    }


    void resnet_forward(float* tensor_A, int height_A, int width_A, int channel_A, int batch,
                        float*& tensor_B, int& height_B, int &width_B, int &channel_B){
        float *tmp_out1,*tmp_out2;
        int height1, width1, channel1;
        int height2, width2, channel2;

        cudaError_t cudaStatus;

        float* A;
        float* B;
        cudaStatus = cudaMalloc((void**)&A, batch * width_A * height_A * channel_A * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            printf("malloc A failed\n");
        }
        cudaStatus = cudaMemcpy((void*)A, (void*)tensor_A, batch * width_A * height_A * channel_A * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            printf("memcpy A failed\n");
        }
//        printf("======= forward begin =======!\n");
        conv1->forward(A, height_A, width_A, channel_A, batch,
                       tmp_out2, height2, width2, channel2);

        cudaFree(A);

//        printf("before maxpooling %d %d %d %d\n",batch, height2, width2, channel2);
        maxpool->forward(tmp_out2, height2, width2, channel2, batch,
                         tmp_out1, height1, width1, channel1);

//        printf("after maxpooling %d %d %d %d \n",
//               batch, channel1,height1,width1);

        cudaFree(tmp_out2);
//        printf("======== stage 1==========\n");
        layer1->forward(tmp_out1, height1, width1, channel1, batch,
                        tmp_out2, height2, width2, channel2);
//        printf("layer1\n");
        cudaFree(tmp_out1);
        layer2->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);
//        printf("layer2\n");
        cudaFree(tmp_out2);
        neck_layer1->forward(tmp_out1, height1, width1, channel1, batch,
                             tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);
        layer3->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);
//        printf("layer3\n");
        cudaFree(tmp_out2);
        neck_layer2->forward(tmp_out1, height1, width1, channel1, batch,
                             tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);
        layer4->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);

        cudaFree(tmp_out2);
        neck_layer3->forward(tmp_out1, height1, width1, channel1, batch,
                             tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);
        layer5->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);

        cudaFree(tmp_out2);
//        printf("========= stage 2 ===========\n");
//        printf("before avg %f %f %d %d %d %d\n",tmp_out1->data[0], tmp_out1->data[2],
//               tmp_out1->batch, tmp_out1->channels,tmp_out1->height,tmp_out1->width);
        avgpool->forward(tmp_out1, height1, width1, channel1, batch,
                         tmp_out2, height2, width2, channel2);
//        printf("after avg: %f %f %d %d %d %d\n",tmp_out2->data[0], tmp_out2->data[2],
//               tmp_out2->batch, tmp_out2->channels,tmp_out2->height,tmp_out2->width);
//        print_tensor<float>(tmp_out2);
        cudaFree(tmp_out1);
        gemm->forward(tmp_out2, height2, width2, channel2, batch,
                      B, height_B, width_B, channel_B);
        cudaFree(tmp_out2);

//        printf("after gemm: %d %d %d %d \n", height_B, width_B, channel_B, batch);
//        tensor_B = (float*)malloc( sizeof(float)*height_B*width_B*channel_B*batch );
        cudaMemcpy((void*)tensor_B, (void*)B, batch * width_B * height_B * channel_B * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(B);
    }
}

extern "C" void init(map<string, float*> Parameters){
       resnet::resnet_init(Parameters);
}

extern "C" void forward(float* tensor_A, int height_A, int width_A, int channel_A, int batch,
                        float*& tensor_B, int& height_B, int &width_B, int &channel_B){
    resnet::resnet_forward(tensor_A, height_A, width_A, channel_A, batch,
                        tensor_B, height_B, width_B, channel_B);
}
