#!/usr/bin/env python
#-*-coding:utf-8-*-
#@File:print_result_onxx.py
import numpy as np
import torch
import onnx
from onnx import numpy_helper
import onnxruntime
import time
import json

# print(onnxruntime.get_device())
# #print(detector.session.get_providers())
# x = torch.randn(1,3,224,224, requires_grad=False)
# #print(x)

# onnx_model = onnx.load(r"resnet18.onnx")
# onnx.checker.check_model(onnx_model)
    
# weights = [] 
# names = []
# model_weights = {}
# for t in onnx_model.graph.initializer:
#     weights.append(numpy_helper.to_array(t).tolist())
#     names.append(t.name)
#     model_weights[t.name] = numpy_helper.to_array(t).tolist()
#     #print(names)
    

# # print(model_weights)
# for k, v in model_weights.items():
#     print(k, np.array(v).shape)
# print('finish print weight!!!!')

# with open('weights.json','w') as file_object:
#     json.dump(model_weights,file_object)

# ort_session = onnxruntime.InferenceSession(r"resnet18.onnx")
# #ort_session.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}])

# start=time.time()

# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# end=time.time()

# ort_out = ort_outs[0]
# #print(ort_out)
# print('time:',end-start)


import onnx
from onnx import numpy_helper
onnx_model   = onnx.load("resnet18.onnx")
INTIALIZERS  = onnx_model.graph.initializer
onnx_weights = {}
for initializer in INTIALIZERS:
    W = numpy_helper.to_array(initializer).tolist()
    onnx_weights[initializer.name] = W

# for k, v in onnx_weights.items():
#     print(k, np.array(v).shape)

# import torchvision.models as tm
# x = tm.resnet18(pretrained=True)
# state_dict = x.state_dict()
# for k, v in state_dict.items():
#     print(k, v.shape)
# torch_out = torch.onnx.export(torch_model, x, onnx_filename, export_params=False) 

# state_dict_list = {k: v.tolist() for k, v in state_dict.items()}
# pretrained = state_dict['conv1.weight']
# onnx = onnx_weights['193']
# print(torch.abs(pretrained-torch.tensor(onnx)).max())
# print(pretrained[0, 0], onnx[0][0])
with open('weights.json','w') as file_object:
    json.dump(onnx_weights, file_object)