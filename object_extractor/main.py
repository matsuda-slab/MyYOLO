# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
import skvideo.io

sys.path.append(os.path.abspath(".."))

import torch
from torchvision import transforms
from model import load_model
from utils.utils import non_max_suppression
from utils.transforms_detect import resize_aspect_ratio

def progress_bar(cur, max):
    i = int((cur / max) * 100)
    bar = ('=' * i) + (' ' * (100 - i))
    print('\r[{}] [{:3.0%}]'.format(bar, cur/max), end='')

weights_path = '../weights/yolov3.weights'
SAVE_DIR     = 'images'
conf_thres   = 0.5
nms_thres    = 0.4
mask_thres   = 150
save_thres   = 20000
INTERVAL     = 120                    # INTERVALフレームに1回抽出する
TARGET_CLASS = [1, 2, 3, 5, 7]      # car, bicycle, motorbike, bus, truck

if sys.argv[1] == None:
    print("Usage : main.py {INPUT VIDEO}")
    sys.exit(1)

cap = skvideo.io.vreader(sys.argv[1])
cap_cv = cv2.VideoCapture(sys.argv[1])
fps = cap_cv.get(cv2.CAP_PROP_FPS)
frames = int(cap_cv.get(cv2.CAP_PROP_FRAME_COUNT))

delay = int(1000.0 / fps)

# グレスケ変換
frame_pre = 0
frame_pre2 = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# モデル生成
model = load_model(weights_path, device, tiny=False, num_classes=80)

frame_cnt = 0
image_cnt = 0
for frame in cap:           # frame は RGB
    detect_flag = 0
    if frame_cnt % INTERVAL == 0:
        frame_cv = frame[:,:,[2,1,0]]       # opencv用に BGR に変換
        orig_h, orig_w = frame.shape[0:2]

        ########################################################################
        # 差分画像用意
        ########################################################################


        # グレスケ変換
        gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)

        # 2フレーム前との差分の絶対値を計算
        diff = cv2.absdiff(gray, frame_pre2)

        # 差分画像を2値化して, マスク画像を算出
        diff[diff < mask_thres] = 0
        diff[mask_thres <= diff] = 255

        diff_back = np.zeros((orig_h, orig_w))

        ########################################################################
        # YOLO を使って車両の写る画像を抽出
        ########################################################################

        # YOLO用前処理
        image = transforms.ToTensor()(frame_cv)
        image = resize_aspect_ratio(image)
        image = torch.from_numpy(image)
        image = image.to(device)
        image = image.permute(2, 0, 1)
        image = image[[2,1,0],:,:]
        image = image.unsqueeze(0)

        model.eval()
        output = model(image)

        output = non_max_suppression(output, conf_thres, nms_thres)
        output = output[0]


        pad_x = max(orig_h - orig_w, 0) * (416 / max(orig_h, orig_w))
        pad_y = max(orig_w - orig_h, 0) * (416 / max(orig_h, orig_w))

        unpad_h = 416 - pad_y
        unpad_w = 416 - pad_x

        output[:, 0] = ((output[:, 0] - pad_x // 2) / unpad_w) * orig_w
        output[:, 1] = ((output[:, 1] - pad_y // 2) / unpad_h) * orig_h
        output[:, 2] = ((output[:, 2] - pad_x // 2) / unpad_w) * orig_w
        output[:, 3] = ((output[:, 3] - pad_y // 2) / unpad_h) * orig_h

        for x_min, y_min, x_max, y_max, conf, class_pred in output:
            if int(class_pred) in TARGET_CLASS:
                if y_min < 0: y_min = 0
                if orig_h < y_max: y_max = orig_h
                if x_min < 0: x_min = 0
                if orig_w < x_max: x_max = orig_w
                # 車両を検出した部分だけ, 差分を有効にする
                for i in range(int(y_min), int(y_max)):
                    for j in range(int(x_min), int(x_max)):
                        diff_back[i][j] = diff[i][j]
                detect_flag = 1

        if detect_flag:
            #cv2.imshow('frame', frame_cv)

            #print("diff.shape :", diff.shape)
            #print("sum(diff_back) :", np.sum(diff_back))
            #cv2.imshow("mask", diff_back)
            if save_thres < np.sum(diff_back):
                image_path = os.path.join(SAVE_DIR, f"{image_cnt:05}.jpg")
                cv2.imwrite(image_path, frame_cv)
                image_cnt = image_cnt + 1

            # 現フレームを1つ前のフレームに設定
            frame_pre2 = frame_pre
            frame_pre = gray

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    frame_cnt = frame_cnt + 1
    progress_bar(frame_cnt, frames)

print('\n')
cv2.destroyAllWindows()
