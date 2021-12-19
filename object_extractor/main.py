# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
import argparse
import skvideo.io
import time

sys.path.append(os.path.abspath(".."))

import torch
from torchvision import transforms
from model import load_model
from utils.utils import non_max_suppression
from utils.transforms_detect import resize_aspect_ratio

import matplotlib.pyplot as plt

def progress_bar(cur, max):
    i = int((cur / max) * 100)
    bar = ('=' * i) + (' ' * (100 - i))
    print('\r[{}] [{:3.0%}]'.format(bar, cur/max), end='')

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('--savedir', default='images')
parser.add_argument('--mask_thres', type=int, default=150)
parser.add_argument('--save_thres', type=int, default=20000)
parser.add_argument('--interval', type=int, default=120)
parser.add_argument('--show', action='store_true', default=False)
parser.add_argument('--nosave', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

weights_path = '../weights/yolov3.weights'
SAVE_DIR     = args.savedir
conf_thres   = 0.5
nms_thres    = 0.4
mask_thres   = 150
save_thres   = 20000
INTERVAL     = 120                    # INTERVALフレームに1回抽出する
TARGET_CLASS = [1, 2, 3, 5, 7]      # car, bicycle, motorbike, bus, truck

if args.video == None:
    print("Usage : main.py {INPUT VIDEO}")
    sys.exit(1)

cap = skvideo.io.vreader(args.video)
cap_cv = cv2.VideoCapture(args.video)
fps = cap_cv.get(cv2.CAP_PROP_FPS)
frames = int(cap_cv.get(cv2.CAP_PROP_FRAME_COUNT))

delay = int(1000.0 / fps)

if not args.nosave:
    os.makedirs(SAVE_DIR, exist_ok=True)

class_names = []
with open('../namefiles/coco.names', 'r') as f:
    class_names = f.read().splitlines()

#colors = [(255, 20, 20), (255, 20, 255), (20, 137, 255), (20, 255, 137), (255, 255, 20)]
cmap = plt.get_cmap('tab20')       # tab20b はカラーマップの種類の1つ
colors = [cmap(i) for i in np.linspace(0, 1, 80)]  # cmap をリスト化 (80分割)

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
    frame_cv = frame[:,:,[2,1,0]]       # opencv用に BGR に変換
    if frame_cnt % INTERVAL == 0:
        start_t = time.time()
        frame_cv_input = frame_cv.copy()       # opencv用に BGR に変換
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
        image = transforms.ToTensor()(frame_cv_input)
        image = resize_aspect_ratio(image)
        image = torch.from_numpy(image)
        image = image.to(device)
        image = image.permute(2, 0, 1)
        image = image[[2,1,0],:,:]
        image = image.unsqueeze(0)
        end_t = time.time()
        if args.debug: print("preprocess : %.4f sec" % (end_t - start_t))

        start_t = time.time()
        model.eval()
        output = model(image)
        end_t = time.time()
        if args.debug: print("inferrence : %.4f sec" % (end_t - start_t))

        start_t = time.time()
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
        end_t = time.time()
        if args.debug: print("postprocess : %.4f sec" % (end_t - start_t))

        start_t = time.time()
        for x_min, y_min, x_max, y_max, conf, class_pred in output:
            if int(class_pred) in TARGET_CLASS:
                if y_min < 0: y_min = 0
                if orig_h < y_max: y_max = orig_h
                if x_min < 0: x_min = 0
                if orig_w < x_max: x_max = orig_w

                box_w = x_max - x_min
                box_h = y_max - y_min
                color = colors[int(class_pred)]
                color = [c * 255 for c in color][0:3]
                color = np.array(color).astype(np.int32).tolist()
                cv2.rectangle(frame_cv_input, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=tuple(color), thickness=2)
                cv2.putText(frame_cv_input, class_names[int(class_pred)], (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=tuple(color), thickness=2)

                # 車両を検出した部分だけ, 差分を有効にする
                for i in range(int(y_min), int(y_max)):
                    for j in range(int(x_min), int(x_max)):
                        diff_back[i][j] = diff[i][j]
                detect_flag = 1
        end_t = time.time()
        if args.debug: print("drawing : %.4f sec" % (end_t - start_t))

        if detect_flag:
            #print("diff.shape :", diff.shape)
            #print("sum(diff_back) :", np.sum(diff_back))
            if args.show:
                start_t = time.time()
                diff_back = cv2.resize(diff_back, dsize=None, fx=0.3, fy=0.3)
                cv2.imshow("mask", diff_back)
                end_t = time.time()
                if args.debug: print("show (mask) : %.4f sec" % (end_t - start_t))
            if save_thres < np.sum(diff_back):
                if args.show:
                    start_t = time.time()
                    frame_cv_input = cv2.resize(frame_cv_input, dsize=None, fx=0.5, fy=0.5)
                    cv2.imshow('frame_saved', frame_cv_input)
                    end_t = time.time()
                    if args.debug: print("show (detected) : %.4f sec" % (end_t - start_t))
                if not args.nosave:
                    image_path = os.path.join(SAVE_DIR, f"{image_cnt:05}.jpg")
                    cv2.imwrite(image_path, frame_cv)
                    image_cnt = image_cnt + 1

            # 現フレームを1つ前のフレームに設定
            frame_pre2 = frame_pre
            frame_pre = gray

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    if args.show:
        start_t = time.time()
        frame_cv = cv2.resize(frame_cv, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow('frame', frame_cv)
        end_t = time.time()
        if args.debug: print("show (original) : %.4f sec" % (end_t - start_t))
    frame_cnt = frame_cnt + 1
    if not args.debug: progress_bar(frame_cnt, frames)

print('\n')
cv2.destroyAllWindows()
