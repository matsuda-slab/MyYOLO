#===============================================================================
# Inferrence on an video or camera
#===============================================================================

from utils.transforms import Resize, DEFAULT_TRANSFORMS
import torch
import os
import sys
import argparse
import numpy as np
import random
from torchvision import transforms
import cv2
from model import load_model
from utils.utils import non_max_suppression

#===============================================================================
# Process arguments
#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='weights/tiny-yolo.model')
parser.add_argument('--video', default='images/car.mp4')
parser.add_argument('--conf_thres', type=float, default=0.5)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--class_names', default='coco.names')
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--nogpu', action='store_true', default=False)
args = parser.parse_args()

#===============================================================================
# Define parameters
#===============================================================================
weights_path = args.weights
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
NUM_CLASSES  = args.num_classes
name_file    = args.class_names
NO_GPU       = args.nogpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NO_GPU or args.quant:
    device = torch.device("cpu")
tensor_type = torch.cuda.FloatTensor
                if (torch.cuda.is_available() and not NO_GPU)
                else torch.FloatTensor
if args.quant:
    tensor_type = torch.ByteTensor

# Load class names from name file
class_names = []
with open(name_file, 'r') as f:
    class_names = f.read().splitlines()

# Create model
model = load_model(weights_path, device, num_classes=NUM_CLASSES,
                    quant=args.quant, jit=True)

# Load video
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print("Can not open video\n")
    sys.exit(1)

while(cap.isOpened()):
    ret, frame = cap.read()
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(416)])((input_image, np.zeros((1,5))))[0].unsqueeze(0)
    image = image.to(device)

    # Inferrence on a frame
    model.eval()
    output = model(image)       # output is in [0-1]

    # Apply NMS
    output = non_max_suppression(output, conf_thres, nms_thres)

    output = output[0]
    print("output.shape :", output.shape)

    # Scale box coordinates according to the original size
    orig_h, orig_w = input_image.shape[0:2]
    pad_x = max(orig_h - orig_w, 0) * (416 / max(orig_h, orig_w))
    pad_y = max(orig_w - orig_h, 0) * (416 / max(orig_h, orig_w))

    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x

    # Restore the original size in consideration of padding
    output[:, 0] = ((output[:, 0] - pad_x // 2) / unpad_w) * orig_w
    output[:, 1] = ((output[:, 1] - pad_y // 2) / unpad_h) * orig_h
    output[:, 2] = ((output[:, 2] - pad_x // 2) / unpad_w) * orig_w
    output[:, 3] = ((output[:, 3] - pad_y // 2) / unpad_h) * orig_h

    # Draw (OpenCV version)
    for x_min, y_min, x_max, y_max, conf, class_pred in output:
        box_w = x_max - x_min
        box_h = y_max - y_min

        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                                                    (0, 0, 255), thickness=2)
        cv2.putText(frame, class_names[int(class_pred)],
                    (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color=(0, 0, 255), thickness=2)

    #cv2.resize(frame, fx=0.5, fy=0.5)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
