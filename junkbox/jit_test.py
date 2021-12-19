import torch
from model import load_model

weights_path = 'weights/yolov3-tiny_doll_light.pt'
device = torch.device("cpu")
model = load_model(weights_path, device, num_classes=2)
model.eval()

#script_model = torch.jit.script(model)
#script_model.save("yolov3-tiny_doll_light_jit.pt")

dummy_input = torch.rand(1, 3, 416, 416)
jit_model = torch.jit.trace(model, dummy_input)
jit_model.save("yolov3-tiny_doll_light_jit.pt")

#######################################################
model = torch.jit.load("yolov3-tiny_doll_light_jit.pt")
