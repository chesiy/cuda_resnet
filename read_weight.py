#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:print_result_onxx.py
import numpy as np
import torch
import onnx
import onnxruntime
import time
import json
from onnx import numpy_helper

# 使用 ONNX 的 API 检查 ONNX 模型
onnx_model = onnx.load(r"resnet18.onnx")
onnx.checker.check_model(onnx_model)

weights = onnx_model.graph.initializer

onnx_weights = {}
for initializer in weights:
    W= numpy_helper.to_array(initializer)
    onnx_weights[initializer.name]=W.tolist()
    print(initializer.name)
print(onnx_weights)

with open("weights.json","w") as f:
    json.dump(onnx_weights,f)
    print("加载入文件完成...")