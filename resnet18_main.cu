#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include "resnet.cu"
#include <map>
#include <json/json.h>
#include <chrono>


#define INPUTSHAPE 3 * 224 * 224
#define OUTPUTSHAPE 1000
#define TESTNUM 10
#define ITERNUM 500

float inputArr[TESTNUM][INPUTSHAPE];
float benchOutArr[TESTNUM][OUTPUTSHAPE];
Resnet18 *resnet18;


void readFileJson(map<string,float*> &parameters)
{
    Json::Reader reader;
    Json::Value root;

    ifstream in("/home/group20/cuda_onnx_python/weights.json",ios::binary);

    if(reader.parse(in,root)){
        Json::Value::Members members = root.getMemberNames();
        for(Json::Value::Members::iterator it = members.begin(); it != members.end(); it++){
            float* para_list = (float*)malloc(sizeof(float) * root[*it].size());
            for (unsigned int i = 0; i < root[*it].size(); i++)
            {
//                cout<<root[*it][i]<<'\n';
                para_list[i] = float(root[*it][i].asDouble());
//                printf("%f ",para_list[i]);
            }
//            printf("\n");
            parameters.insert(pair<string, float*>(*it, para_list));
        }
    }

    in.close();
}


void readInput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++)
            fscanf(fp, "%f", &inputArr[i][j]);
}

void readOutput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%f", &benchOutArr[i][j]);
}

void checkOutput(float *out1, float *out2)
{
    float maxDiff = 0;
    for (int i = 0; i < OUTPUTSHAPE; i++)
    {
        maxDiff = (fabs(out1[i] - out2[i]) > maxDiff) ? fabs(out1[i] - out2[i]) : maxDiff;
//        if(fabs(out1[i]-out2[i]) > 1e-5){
//            printf("%f ",fabs(out1[i]-out2[i]));
//        }
    }
    if (maxDiff > 1e-5)
    {
        printf("Output dismatch. MaxDiff is %.7f\n", maxDiff);
//        exit(-1);
    }
}

// TODO: 读取权重
void initModel(){
	map<string, float*> parameters;
    readFileJson(parameters);
	resnet18 = new Resnet18{parameters};
}

// TODO: 实现自己的inference
void inference(float *input, float *output){
	int height_B,width_B,channel_B;
	//printf("%f",sizeof(input)/sizeof(float));
	resnet18->forward(input, 224, 224, 3, 1, *&output, height_B, width_B, channel_B);

}

int main()
{
    initModel(); // 读取网络权重
    
    readInput("./resnet18Input.txt");   // 读取输入
    readOutput("./resnet18Output.txt"); // 读取标准输出
    float sumTime = 0;
    for (int i = 0; i < TESTNUM; i++)
    {
        float inferOut[1000];
        for (int j = 0; j < ITERNUM; j++)
        {
            float Onetime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            // 执行Inference
            inference(inputArr[i], inferOut);
            checkOutput(benchOutArr[i], inferOut);

            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Onetime, start, stop);
            // 累加单次推理消耗时间
            sumTime += Onetime;
        }
        checkOutput(benchOutArr[i], inferOut);
    }
    printf("Average Time is: %f\n", (sumTime / TESTNUM / ITERNUM));
}