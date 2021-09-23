import torch
import torch.nn as nn
import torchvision
import pickle
import numpy as np
from model import YOLO, load_model, YOLOLayer
#from models import Darknet

weights_path = 'weights/yolov3-tiny.weights'
with open(weights_path, "rb") as f:
    weights = np.fromfile(f, dtype=np.float32)

print(weights[0:1000])
