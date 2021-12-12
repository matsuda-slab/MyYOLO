import torch
from model import load_model

weights_path = 'weights/yolov3-tiny.weights'
device = torch.device("cpu")

model = load_model(weights_path, device, 80)
torch.save(model.state_dict(), 'weights/yolov3-tiny_converted.pt')
