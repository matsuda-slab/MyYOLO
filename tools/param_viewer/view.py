import torch

params = torch.load('../../weights/yolov3-tiny_car1_sep.pt', map_location='cpu')

print(params.keys())
print(params['conv2.conv_dw.weight'].shape)
print(params['conv2.conv_pw.weight'].shape)
