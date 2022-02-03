import numpy as np
import torch
import collections

conv_dw_w = torch.randn(3, 1, 3, 3)
conv_dw_b = torch.randn(3)
bn_dw_w   = torch.randn(3)
bn_dw_b   = torch.randn(3)
bn_dw_rm  = torch.randn(3)
bn_dw_rv  = torch.from_numpy(np.random.rand(3))
conv_pw_w = torch.randn(16, 3, 1, 1)
conv_pw_b = torch.randn(16)
bn_pw_w   = torch.randn(16)
bn_pw_b   = torch.randn(16)
bn_pw_rm  = torch.randn(16)
bn_pw_rv  = torch.from_numpy(np.random.rand(16))

print(bn_dw_rv)
print(bn_pw_rv)

dic = collections.OrderedDict()
dic['conv1.conv_dw.weight']     = conv_dw_w
dic['conv1.conv_dw.bias']       = conv_dw_b
dic['conv1.bn_dw.weight']       = bn_dw_w
dic['conv1.bn_dw.bias']         = bn_dw_b
dic['conv1.bn_dw.running_mean'] = bn_dw_rm
dic['conv1.bn_dw.running_var']  = bn_dw_rv
dic['conv1.conv_pw.weight']     = conv_pw_w
dic['conv1.conv_pw.bias']       = conv_pw_b
dic['conv1.bn_pw.weight']       = bn_pw_w
dic['conv1.bn_pw.bias']         = bn_pw_b
dic['conv1.bn_pw.running_mean'] = bn_pw_rm
dic['conv1.bn_pw.running_var']  = bn_pw_rv

torch.save(dic, "weights/nomerge.pt")
