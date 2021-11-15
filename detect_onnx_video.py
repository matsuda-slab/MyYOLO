import onnx_tf.backend
import onnx
import tensorflow as tf
from utils.transforms_detect import resize_aspect_ratio
import torch
import os
import sys
import argparse
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
#from PIL import Image
import cv2
from utils.utils import non_max_suppression
import time

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='yolo-tiny_car.onnx')
parser.add_argument('--video', default='images/car.mp4')
parser.add_argument('--conf_thres', type=float, default=0.5)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--output_image', default='output.jpg')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--class_names', default='coco.names')
args = parser.parse_args()

weights_path = args.weights
video_path   = args.video
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
output_path  = args.output_image
NUM_CLASSES  = args.num_classes
name_file    = args.class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# クラスファイルからクラス名を読み込む
class_names = []
with open(name_file, 'r') as f:
    class_names = f.read().splitlines()

# モデルファイルからモデルを読み込む
model = onnx.load(weights_path)

tf_model = onnx_tf.backend.prepare(model, device='CPU')

# 動画の読み込み
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Can not open video\n")
    sys.exit(1)

while(cap.isOpened()):
    start = time.time();
    ret, frame = cap.read()
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_load_t = time.time()
#print("### input_image.shape :", input_image.shape)
    input_image = input_image[:, :, :] / 255.0
    image = resize_aspect_ratio(input_image, use_torch=False)      # by opencv
    image = image.transpose(2, 0, 1)
    image = image[np.newaxis, [2,1,0], :, :]
    image = tf.cast(image, tf.float32)

    # 入力画像からモデルによる推論を実行する
    output = tf_model.run(image)       # 出力座標は 0~1 の値

    # 推論結果に NMS をかける
    # ここの outputの出力座標 はすでに 0~416 にスケールされている
    output_torch = torch.from_numpy(output[0])
    output = non_max_suppression(output_torch, conf_thres, nms_thres)

    nms_boxes = output[0]
    print("output.shape :", nms_boxes.shape)

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
    unpad_boxes = np.zeros((nms_boxes.shape[0], 4))
    unpad_boxes[:, 0] = ((nms_boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    unpad_boxes[:, 1] = ((nms_boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    unpad_boxes[:, 2] = ((nms_boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    unpad_boxes[:, 3] = ((nms_boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    # 描画 opencv バージョン
    for x_min, y_min, x_max, y_max, conf, class_pred in unpad_boxes:
        box_w = x_max - x_min
        box_h = y_max - y_min

        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), thickness=2)
        cv2.putText(frame, 'car', (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 255), thickness=2)

    #cv2.resize(frame, fx=0.5, fy=0.5)
    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    end = time.time()
    print("elapsed time = %.4f sec" % (end - start))

cap.release()
cv2.destroyAllWindows()

