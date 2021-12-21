#include <string>
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "resnet.cu"
//#include "tensor.cu"
#include <map>
#include <chrono>

using namespace std;

void matgen(float* a, int x, int y)
{
    int i, j;
    for (i = 0; i < x; i++)
    {
        for (j = 0; j < y; j++)
        {
//            a[i * y + j] = (float)rand() / RAND_MAX + (float)rand()*2 / (RAND_MAX);
            a[i*y+j] = (i*y+j)*1.0/1000;
        }
    }
}

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

int main(){
    int x = 224*1;
    int y = 224*3;

    float *M = (float*)malloc(sizeof(float)* x * y);

    srand(0);
    matgen(M, x, y);			//产生矩阵M


    float* B;
    int height_B,width_B,channel_B;
    map<string, float*> parameters;
    readFileJson(parameters);

    Resnet18 *resnet18 = new Resnet18{parameters};

    chrono::milliseconds ms = chrono::duration_cast< chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
    long start= ms.count();

    resnet18->forward(M, 224, 224, 3, 1,
                      B, height_B, width_B, channel_B);

    ms = chrono::duration_cast< chrono::milliseconds >(chrono::system_clock::now().time_since_epoch());
    long end= ms.count();

    printf("time: %ld\n", end-start);

    printf("B: %d %d %d %d\n",height_B,width_B,channel_B, 1);
    print_tensor(B, 1, channel_B, height_B, width_B);

    return 0;
}