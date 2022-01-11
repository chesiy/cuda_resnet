#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "block.cu"
#include "string.h"
#include "stdio.h"
#include <map>

/*
basic block, input shape torch.Size([1, 64, 56, 56]), output shape torch.Size([1, 64, 56, 56])
basic block, input shape torch.Size([1, 64, 56, 56]), output shape torch.Size([1, 64, 56, 56])
layer1 finish
basic block, input shape torch.Size([1, 64, 56, 56]), output shape torch.Size([1, 128, 28, 28])
basic block, input shape torch.Size([1, 128, 28, 28]), output shape torch.Size([1, 128, 28, 28])
layer2 finish
basic block, input shape torch.Size([1, 128, 28, 28]), output shape torch.Size([1, 256, 14, 14])
basic block, input shape torch.Size([1, 256, 14, 14]), output shape torch.Size([1, 256, 14, 14])
layer3 finish
basic block, input shape torch.Size([1, 256, 14, 14]), output shape torch.Size([1, 512, 7, 7])
basic block, input shape torch.Size([1, 512, 7, 7]), output shape torch.Size([1, 512, 7, 7])
layer4 finish
*/


/*
TOO LAZY TO USE ENUMERATE
conv_type:
    0: im2col, pre-allocate weight
    1: im2col, allocate weight in forward
    2: winograd 4x4, pre-allocate weight,
    3: winograd 4x4 + relu, pre-allocate weight,
    4: im2col + relu, pre-allocate weight
    5: im2col + add + relu, pre-allocate weight,
    7: im2col + avg_pooling, pre-allocate weight,
    -1: im2col, pre-allocate weight, only for 7x7
*/


namespace resnet{
    conv_im2col* conv_first;
    maxpooling2d *maxpool;
    GlobalAvgpooling *avgpool;
    Gemm *gemm;
    BasicBlock *layer1,*layer2,*layer3,*layer4,*layer5;
    Bottleneck *neck_layer1,*neck_layer2,*neck_layer3;

    float* Input;

    void resnet_init(map<string, float*> Parameters, int batch_size, int in_channel, int in_height, int in_width){
        conv_first = new conv_im2col{3,64,Parameters["193"],Parameters["194"], -1, 7, 3, 2, 224, 1};

        maxpool = new maxpooling2d{3,1,2};
        layer1 = new BasicBlock{64,64,Parameters["196"],Parameters["197"],Parameters["199"],Parameters["200"], 3, 6, 56};
        // layer1 = new BasicBlock{64,64,Parameters["196"],Parameters["197"],Parameters["199"],Parameters["200"], 4, 5, 56};
        layer2 = new BasicBlock{64,64,Parameters["202"],Parameters["203"],Parameters["205"],Parameters["206"], 3, 6, 56};
        // layer2 = new BasicBlock{64,64,Parameters["202"],Parameters["203"],Parameters["205"],Parameters["206"], 4, 5, 56};

        neck_layer1 = new Bottleneck{64,128,Parameters["208"],Parameters["209"],Parameters["211"],Parameters["212"],Parameters["214"],Parameters["215"], 2, 4, 2, 5, 56};
        layer3 = new BasicBlock{128,128,Parameters["217"],Parameters["218"],Parameters["220"],Parameters["221"], 3, 6, 28};

        neck_layer2 = new Bottleneck{128,256,Parameters["223"],Parameters["224"],Parameters["226"],Parameters["227"],Parameters["229"],Parameters["230"], 2, 4, 2, 5, 28};
        layer4 = new BasicBlock{256,256,Parameters["232"],Parameters["233"],Parameters["235"],Parameters["236"], 3, 6, 14};

        neck_layer3 = new Bottleneck{256,512,Parameters["238"],Parameters["239"],Parameters["241"],Parameters["242"],Parameters["244"],Parameters["245"], 2, 4, 2, 5, 14};
        layer5 = new BasicBlock{512,512,Parameters["247"],Parameters["248"],Parameters["250"],Parameters["251"], 3, 6, 7};

        avgpool = new GlobalAvgpooling{};
        gemm = new Gemm{1, 512, 1000, Parameters["fc.weight"],Parameters["fc.bias"]};

        cudaError_t cudaStatus;
        
        cudaStatus = cudaMalloc((void**)&Input, 1*224*224*3 * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            printf("malloc Input failed\n");
        }
    }


    void resnet_forward(float* tensor_A, int height_A, int width_A, int channel_A, int batch,
                        float*& tensor_B, int& height_B, int &width_B, int &channel_B){
        float *tmp_out1,*tmp_out2, *tmp_out0;
        batch = 1;
        height_B = 1;
        width_B = 1;
        channel_B = 1000;

        cudaError_t cudaStatus;
        // float* A;
        float* B;

        // cudaStatus = cudaMalloc((void**)&Input, batch * height_A * width_A * 3 * sizeof(float));
        // check_status(cudaStatus, "input memory malloc");

        cudaStatus = cudaMemcpy((void*)Input, (void*)tensor_A, 1*224*224*3 * sizeof(float), cudaMemcpyHostToDevice);
        check_status(cudaStatus, "input memory copy");

        conv_first->forward(Input, tmp_out2);
        // cudaFree(Input);

        maxpool->forward(tmp_out2, 112, 112, 64, 1,
                         tmp_out0, 56, 56, 64);

        // cudaFree(tmp_out2);
        layer1->forward(tmp_out0, tmp_out2);

        layer2->forward(tmp_out2, tmp_out1);

        neck_layer1->forward(tmp_out1, tmp_out2);

        layer3->forward(tmp_out2, tmp_out1);

        neck_layer2->forward(tmp_out1, tmp_out2);

        layer4->forward(tmp_out2, tmp_out1);

        neck_layer3->forward(tmp_out1, tmp_out2);

        layer5->forward(tmp_out2, tmp_out1);

//        printf("========= stage 2 ===========\n");
        avgpool->forward(tmp_out1, 7, 7, 512, 1,
                         tmp_out2, 1, 1, 512);
                         
        gemm->forward(tmp_out2, B);
        // cudaFree(tmp_out0);
        // cudaFree(tmp_out2);

        cudaMemcpy((void*)tensor_B, (void*)B, batch * width_B * height_B * channel_B * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaFree(B);
        // printf("success for one time");
    }
}

extern "C" void init(map<string, float*> Parameters){
       resnet::resnet_init(Parameters, 1, 3, 224, 224);
}

extern "C" void forward(float* tensor_A, int height_A, int width_A, int channel_A, int batch,
                        float*& tensor_B, int& height_B, int &width_B, int &channel_B){
    resnet::resnet_forward(tensor_A, height_A, width_A, channel_A, batch,
                        tensor_B, height_B, width_B, channel_B);
}
