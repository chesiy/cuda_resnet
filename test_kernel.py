import torch
import torch.nn as nn

# batch_size = 1
# in_channels = 2
# out_channels = 1
# kernel_size = 3
# inp_row = 2
# inp_col = 2

batch_size = 2
in_channels = 2
out_channels = 4
kernel_size = 3
inp_row = 7
inp_col = 7

# batch_size = 2
# in_channels = 2
# out_channels = 4
# kernel_size = 3
# inp_row = 6
# inp_col = 6

conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=1, bias=True)

kernel = torch.zeros((out_channels*in_channels*kernel_size*kernel_size))
for i in range(len(kernel)):
    kernel[i] = i

kernel = kernel.reshape(out_channels, in_channels, kernel_size, kernel_size)

inp = torch.zeros((batch_size*in_channels*inp_row*inp_col))
for i in range(len(inp)):
    inp[i] = i

bias = torch.zeros(out_channels)

# for i in range(len(bias)):
#     bias[i] = i

bias = bias.reshape(out_channels)

inp = inp.reshape(batch_size, in_channels, inp_row, inp_col)

# for k, v in conv.named_parameters().items():
# for k, v in conv.state_dict().items():
#     print(k, v)

kernel_dict = {'weight': kernel, 'bias': bias}
conv.load_state_dict(kernel_dict)

out = conv(inp)
print(out)