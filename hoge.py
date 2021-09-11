import torch
import torch.nn as nn
import torchvision

weights_path = 'yolov3-tiny.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(weights_path, map_location=device)
