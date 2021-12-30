Group20 Resnet18
=============================
组员：卞思沅 陈思远 林端儿

解析onnx模型并获取参数
-----------------------------
1、通过get_onnx_weight.py获得weight.json
```
conda activate onnx_env
python get_onnx_weight.py
```

2、通过[Jsoncpp](https://github.com/open-source-parsers/jsoncpp)加载参数至模型中，其中/json以及/json_lib即为jsoncpp所需文件

Baseline
------------------------------
通过pytorch搭建Resnet18并将backend设为cudnn作为baseline
```
cd /home/group20/git/resnet_python/try_resnet_format.py
conda activate onnx_env
python try_resnet_format.py
```

CUDA搭建与实现
------------------------------
kernels.cu: MaxPooing AvgPooling Relu Add MaxMul

winograd: conv_winograd_4x4_3x3.cu conv_winograd_gpu.cu

im2col: conv_im2col.cu

resnet_extern.cu: resnet

resnet18_main.cc: main

```
make
./hello
```

输入输出文件及模型
----------------------------
resnet18Input.txt

resnet18Output.txt

resnet18.onnx

实验结果
---------------------------
|    Methods    |    time(ms)    |
|:------:|:-----:|
|Baseline (pytorch)|2.6|
|CPU (pytorch)|   |
|winograd|   |
|im2col|     |
wingrad+im2col|   |
