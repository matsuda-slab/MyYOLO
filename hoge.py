import torch
import torch.nn as nn
import torchvision
import pickle
import numpy as np
from model import YOLO, load_model, YOLOLayer
from models import Darknet

weights_path = 'weights/yolov3-tiny.weights'
#weights_path = 'results/20210913_150240/yolo-tiny.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
<<<<<<< HEAD
ckpt = torch.load(weights_path, map_location=torch.device('cpu'))

#num_classes = 1
#ylch = (5 + num_classes) * 3
#
#update_param_names = ['conv10.weight', 'conv10.bias',
#                      'conv13.weight', 'conv13.bias']
#
#
#model, param_to_update = load_model(weights_path, device, trans=True)
#state_dict = model.state_dict()
##print(model.conv10.weight)
#model.conv10 = nn.Conv2d( 512, ylch, kernel_size=1, stride=1, padding=0, bias=1)
#model.conv13 = nn.Conv2d( 256, ylch, kernel_size=1, stride=1, padding=0, bias=1)
#model.yolo1 = YOLOLayer(model.anchors[1], model.img_size, num_classes)
#model.yolo2 = YOLOLayer(model.anchors[0], model.img_size, num_classes)
##print(model.conv10.weight)
#
##print(type(model.state_dict()))
##print(model.state_dict())
#for (key, param) in model.named_parameters():
#    print(key)
#    if key in update_param_names:
#        param.requires_grad = True
#        param_to_update.append(param)
#    else:
#        param.requires_grad = False
#
#print(param_to_update)
=======
#model = Darknet('yolov3-tiny.cfg')
#model.load_darknet_weights(weights_path)

model = load_model(weights_path, device)
print("weights :", model.conv1.weight)

weights_path = 'weights/yolov3-tiny.pt'
model = load_model(weights_path, device, num_classes=1, trans=True)
print("pt :", model.conv1.weight)
>>>>>>> 0d914a25ed43301f57aea54fcd13eb89cd2657b7
