import sys
import torch

params = torch.load(sys.argv[1], map_location='cpu')

for key in params.keys():
    print(key)

#print("conv1.bn_pw.weight:", params['conv1.bn_pw.weight'])
#print("conv1.bn_pw.bias:", params['conv1.bn_pw.bias'])
#print("conv1.bn_pw.running_mean:", params['conv1.bn_pw.running_mean'])
#print("conv1.bn_pw.running_var:", params['conv1.bn_pw.running_var'])
#print("conv1.conv_dw.weight:", params['conv1.conv_dw.weight'])
#print("conv1.bn_dw.weight:", params['conv1.bn_dw.weight'])
#print("conv1.bn_dw.bias:", params['conv1.bn_dw.bias'])
#print("conv1.bn_dw.running_mean:", params['conv1.bn_dw.running_mean'])
#print("conv1.bn_dw.running_var:", params['conv1.bn_dw.running_var'])
