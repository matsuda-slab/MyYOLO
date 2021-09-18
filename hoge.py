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
#model = Darknet('yolov3-tiny.cfg')
#model.load_darknet_weights(weights_path)

model = load_model(weights_path, device)
print("weights :", model.conv1.weight)

weights_path = 'weights/yolov3-tiny.pt'
model = load_model(weights_path, device, num_classes=1, trans=True)
print("pt :", model.conv1.weight)
