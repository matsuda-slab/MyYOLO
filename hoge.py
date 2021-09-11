import torch
import torch.nn as nn
import torchvision
import pickle
import numpy as np

weights_path = 'weights/yolov3-tiny.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(weights_path, map_location=torch.device('cpu'))
print("pretrained weights :")
print(ckpt['model']['module_list.0.Conv2d.weight'])

# with open(weights_path, "rb") as f:
#     param = pickle.load(str(f))

from model import YOLO, load_model

print("original weights :")
model = load_model(weights_path, device)
print(model.state_dict()['conv1.weight'])
