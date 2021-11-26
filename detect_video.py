from utils.transforms import Resize, DEFAULT_TRANSFORMS
import torch
import os
import sys
import argparse
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from model import YOLO, load_model
from utils.utils import non_max_suppression

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

weights_path = args.weights
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
NUM_CLASSES  = args.num_classes
name_file    = args.class_names
NO_GPU       = args.nogpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NO_GPU or args.quant:
    device = torch.device("cpu")
tensor_type = torch.cuda.FloatTensor if (torch.cuda.is_available() and not NO_GPU) else torch.FloatTensor
if args.quant:
    tensor_type = torch.ByteTensor

# クラスファイルからクラス名を読み込む
class_names = []
with open(name_file, 'r') as f:
    class_names = f.read().splitlines()

# モデルファイルからモデルを読み込む
model = load_model(weights_path, device, num_classes=NUM_CLASSES, quant=args.quant, jit=True)

# 動画の読み込み
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

    # 入力画像からモデルによる推論を実行する
    model.eval()
    output = model(image)       # 出力座標は 0~1 の値

    # 推論結果に NMS をかける
    # ここの outputの出力座標 はすでに 0~416 にスケールされている
    output = non_max_suppression(output, conf_thres, nms_thres)

    output = output[0]
    print("output.shape :", output.shape)

    ### 推論結果のボックスの位置(0~1)を元画像のサイズに合わせてスケールする
    orig_h, orig_w = input_image.shape[0:2]
    # 416 x 416 に圧縮したときに加えた情報量 (?) を算出
    # 例えば, 640 x 480 を 416 x 416 にリサイズすると, 横の長さに合わせると
    # 縦が 416 より小さくなってしまうので, y成分に情報を加えて, 416にしている
    # と思われる. そのため, ここで加えた余分な情報を取り除く(量を決めるための)
    # unpad_h, unpad_w を算出している
    pad_x = max(orig_h - orig_w, 0) * (416 / max(orig_h, orig_w))
    pad_y = max(orig_w - orig_h, 0) * (416 / max(orig_h, orig_w))

    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x

    # 追加した(paddingした)情報を考慮した, 元画像サイズへの復元
    # (ここの式よくわからない)
    output[:, 0] = ((output[:, 0] - pad_x // 2) / unpad_w) * orig_w
    output[:, 1] = ((output[:, 1] - pad_y // 2) / unpad_h) * orig_h
    output[:, 2] = ((output[:, 2] - pad_x // 2) / unpad_w) * orig_w
    output[:, 3] = ((output[:, 3] - pad_y // 2) / unpad_h) * orig_h

    # 描画 opencv バージョン
    for x_min, y_min, x_max, y_max, conf, class_pred in output:
        box_w = x_max - x_min
        box_h = y_max - y_min

        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), thickness=2)
        cv2.putText(frame, 'car', (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 255), thickness=2)

    #cv2.resize(frame, fx=0.5, fy=0.5)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
