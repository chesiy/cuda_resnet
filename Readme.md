# Group20 Resnet18
组员：卞思沅 陈思远 林端儿

*In this repo, we reimplement **ResNet18** with CUDA, and make optimizations to speed up inference. Experiments show that our code is faster than the pytorch version (batchsize=1).*

*We use im2col and winograd(4x4) to speed up convolution*.

onnx模型下载链接：链接：https://pan.baidu.com/s/1eVvb2OedbnYR7m6PG-U_Cw   提取码：ksm8

## 解析onnx模型并获取参数
1、通过get_onnx_weight.py获得weight.json

```
cd /home/group20/cuda_onnx_python/
conda activate onnx_env
python get_onnx_weight.py
```

2、通过[Jsoncpp](https://github.com/open-source-parsers/jsoncpp)加载参数至模型中，其中/json以及/json_lib即为jsoncpp所需文件

## Baseline
通过pytorch搭建Resnet18并将backend设为cudnn作为baseline
```
cd /home/group20/git/resnet_python/try_resnet_format.py
conda activate onnx_env
python try_resnet_format.py
```

## CUDA搭建与实现
kernels.cu: MaxPooing AvgPooling Relu Add

GEMM: matmul.cu

winograd: conv_winograd_4x4_3x3.cu conv_winograd_gpu.cu

im2col: conv_im2col.cu

resnet_extern.cu: resnet

resnet18_main.cc: main

```
cd /home/group20/resnet_cuda/tmp/final_version/
make
./hello
```

## 输入输出文件及模型

resnet18Input.txt

resnet18Output.txt

resnet18.onnx

weight.json

## 实验结果

|      Methods       | time(ms) |
| :----------------: | :------: |
| Baseline (pytorch) |   2.67    |
|     our model      |   2.26    |
