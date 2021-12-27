#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "block.cu"
#include "string.h"
#include <iostream>
#include "stdio.h"
#include <map>

using namespace std;

void print_tensor(float* Ts, int batch, int channels, int height, int width){
    for(int i=0;i<batch;i++){
        for(int j=0;j<channels;j++){
            for(int k=0;k<height;k++){
                for(int t=0;t<width;t++){
                    printf("%f ",Ts[i*(channels*width*height)+j*(width*height)+k*width+t]);
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

class Resnet18{
private:
    conv_im2col *conv1;
    Relu *relu;
    maxpooling2d *maxpool;
    GlobalAvgpooling *avgpool;
    Gemm *gemm;
    BasicBlock *layer1,*layer2,*layer3,*layer4,*layer5;
    Bottleneck *neck_layer1,*neck_layer2,*neck_layer3;
    map<string, float*> Parameters;
    cudaStream_t stream1;

public:
    Resnet18(map<string, float*> param):Parameters(param){
//        cudaStreamCreate(&stream1);
        cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
        conv1 = new conv_im2col{3,64,Parameters["193"],Parameters["194"], NULL, 7,1,3,2};
        relu = new Relu{};
        maxpool = new maxpooling2d{3,1,2};
        layer1 = new BasicBlock{64,64,Parameters["196"],Parameters["197"],Parameters["199"],Parameters["200"],4};
        layer2 = new BasicBlock{64,64,Parameters["202"],Parameters["203"],Parameters["205"],Parameters["206"],4};
        neck_layer1 = new Bottleneck{64,128,Parameters["208"],Parameters["209"],Parameters["211"],Parameters["212"],Parameters["214"],Parameters["215"],&stream1,2,4};
        layer3 = new BasicBlock{128,128,Parameters["217"],Parameters["218"],Parameters["220"],Parameters["221"],4};
        neck_layer2 = new Bottleneck{128,256,Parameters["223"],Parameters["224"],Parameters["226"],Parameters["227"],Parameters["229"],Parameters["230"],&stream1,2,4};
        layer4 = new BasicBlock{256,256,Parameters["232"],Parameters["233"],Parameters["235"],Parameters["236"],4};
        neck_layer3 =new Bottleneck{256,512,Parameters["238"],Parameters["239"],Parameters["241"],Parameters["242"],Parameters["244"],Parameters["245"],&stream1,2,2};
        layer5 = new BasicBlock{512,512,Parameters["247"],Parameters["248"],Parameters["250"],Parameters["251"],2};
        // all the conv use im2col
//        layer1 = new BasicBlock{64,64,Parameters["196"],Parameters["197"],Parameters["199"],Parameters["200"],1};
//        layer2 = new BasicBlock{64,64,Parameters["202"],Parameters["203"],Parameters["205"],Parameters["206"],1};
//        neck_layer1 = new Bottleneck{64,128,Parameters["208"],Parameters["209"],Parameters["211"],Parameters["212"],Parameters["214"],Parameters["215"],&stream1,2,1};
//        layer3 = new BasicBlock{128,128,Parameters["217"],Parameters["218"],Parameters["220"],Parameters["221"],1};
//        neck_layer2 = new Bottleneck{128,256,Parameters["223"],Parameters["224"],Parameters["226"],Parameters["227"],Parameters["229"],Parameters["230"],&stream1,2,1};
//        layer4 = new BasicBlock{256,256,Parameters["232"],Parameters["233"],Parameters["235"],Parameters["236"],1};
//        neck_layer3 =new Bottleneck{256,512,Parameters["238"],Parameters["239"],Parameters["241"],Parameters["242"],Parameters["244"],Parameters["245"],&stream1,2,1};
//        layer5 = new BasicBlock{512,512,Parameters["247"],Parameters["248"],Parameters["250"],Parameters["251"],1};
//        avgpool = new GlobalAvgpooling{};
//        gemm = new Gemm{512,1000,Parameters["fc.weight"],Parameters["fc.bias"]};

    }

    void forward(float* tensor_A, int height_A, int width_A, int channel_A, int batch,
                 float*& tensor_B, int& height_B, int &width_B, int &channel_B){
        float *tmp_out1,*tmp_out2;
        int height1, width1, channel1;
        int height2, width2, channel2;

//        float Onetime;
//        cudaEvent_t start, stop;
//        cudaEventCreate(&start);
//        cudaEventCreate(&stop);
//        cudaEventRecord(start, 0);

        float* A;
        float* B;
        cudaMalloc((void**)&A, batch * width_A * height_A * channel_A * sizeof(float));
        cudaMemcpy((void*)A, (void*)tensor_A, batch * width_A * height_A * channel_A * sizeof(float), cudaMemcpyHostToDevice);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("Memcpy time: %f\n", Onetime);

//        cudaEventRecord(start, 0);

        conv1->forward(A, height_A, width_A, channel_A, batch,
                       tmp_out1, height1, width1, channel1);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("conv1 time: %f\n", Onetime);

//        cudaEventRecord(start, 0);

        relu->forward(tmp_out1, height1, width1, channel1, batch,
                      tmp_out2, height2, width2, channel2);

        cudaFree(tmp_out1);
        maxpool->forward(tmp_out2, height2, width2, channel2, batch,
                         tmp_out1, height1, width1, channel1);
        cudaFree(tmp_out2);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("relu+maxpooling time: %f\n", Onetime);

//        cudaEventRecord(start, 0);

        layer1->forward(tmp_out1, height1, width1, channel1, batch,
                        tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("basic block1 time: %f\n", Onetime);

//        cudaEventRecord(start, 0);

        layer2->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);
        cudaFree(tmp_out2);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("basic block2 time: %f\n", Onetime);

//        cudaEventRecord(start, 0);

        neck_layer1->forward(tmp_out1, height1, width1, channel1, batch,
                             tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("bottleneck1 time: %f\n", Onetime);
//
//        cudaEventRecord(start, 0);

        layer3->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);
//        printf("layer3\n");
        cudaFree(tmp_out2);
        neck_layer2->forward(tmp_out1, height1, width1, channel1, batch,
                             tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);
        layer4->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);
//        printf("layer4\n");
        cudaFree(tmp_out2);
        neck_layer3->forward(tmp_out1, height1, width1, channel1, batch,
                             tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);
        layer5->forward(tmp_out2, height2, width2, channel2, batch,
                        tmp_out1, height1, width1, channel1);

        cudaFree(tmp_out2);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("layer3 4 5 time: %f\n", Onetime);

//        cudaEventRecord(start, 0);

        avgpool->forward(tmp_out1, height1, width1, channel1, batch,
                         tmp_out2, height2, width2, channel2);
        cudaFree(tmp_out1);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("avgpooling time: %f\n", Onetime);

//        cudaEventRecord(start, 0);

        gemm->forward(tmp_out2, height2, width2, channel2, batch,
                      B, height_B, width_B, channel_B);

//        cudaDeviceSynchronize();
//        cudaEventRecord(stop, 0);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&Onetime, start, stop);
//        printf("gemm time: %f\n", Onetime);

        tensor_B = (float*)malloc( sizeof(float)*height_B*width_B*channel_B*batch );
        cudaMemcpy((void*)tensor_B, (void*)B, batch * width_B * height_B * channel_B * sizeof(float), cudaMemcpyDeviceToHost);

    }

};


