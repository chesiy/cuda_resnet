from resnet import ResNet
import torch
import numpy as np
import json
import copy


init_dict = {'193':'conv1.weight', '196':'layer1.0.conv1.weight',
        '199':'layer1.0.conv2.weight', '202':'layer1.1.conv1.weight',
        '205':'layer1.1.conv2.weight', '208':'layer2.0.conv1.weight',
        '211':'layer2.0.conv2.weight', '214':'layer2.0.downsample.0.weight',
        '217':'layer2.1.conv1.weight', '220':'layer2.1.conv2.weight',
        '223':'layer3.0.conv1.weight', '226':'layer3.0.conv2.weight',
        '229':'layer3.0.downsample.0.weight', '232':'layer3.1.conv1.weight',
        '235':'layer3.1.conv2.weight', '238':'layer4.0.conv1.weight',
        '241':'layer4.0.conv2.weight', '244':'layer4.0.downsample.0.weight',
        '247':'layer4.1.conv1.weight', '250':'layer4.1.conv2.weight',
        'fc.weight':'fc.weight', 'fc.bias':'fc.bias'}

appended_dict = copy.deepcopy(init_dict)

for k, v in init_dict.items():
    layer_type = v.split('.')[-2]
    if layer_type[:4] == 'conv' or layer_type == '0':
        bias_name = '.'.join(v.split('.')[:-1] + ['bias'])
        init_name = str(int(k) + 1)
        appended_dict[init_name] = bias_name

# print(appended_dict)
inverse_dict = {v: k for k, v in appended_dict.items()}
with open('weights.json', 'r') as f:
    new_weight = json.load(f)
# new_weight = json.loads('weights.json')
m = ResNet("resnet18")

res_state_dict = m.state_dict()
# for k, v in res_state_dict.items():
#     print(k, v.shape)

new_state_dict = {k: torch.tensor(new_weight[inverse_dict[k]]) for k, _ in res_state_dict.items()}

for k, v in appended_dict.items():
    if not v in new_state_dict:
        print(k, v)

m.load_state_dict(new_state_dict)

x = torch.zeros(1*3*24*24)
for i in range(1*3*24*24):
    x[i]=i*1.0 / 1000.0

x = x.reshape((1,3,24,24))
out = m(x)
# print(out)