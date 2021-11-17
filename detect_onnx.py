#===============================================================================
# onnxモデル (hoge.pt) を読み込み,
# tensorflow バックエンドで推論する
#===============================================================================

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
#import torch.nn as nn
#import torch.nn.functional as F
import cv2
from utils.utils_tf import non_max_suppression
import time

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='weights/yolo-tiny.onnx')
parser.add_argument('--image', default='images/dog.jpg')
parser.add_argument('--conf_thres', type=float, default=0.5)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--output_image', default='output.jpg')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--class_names', default='coco.names')
parser.add_argument('--nogpu', action='store_true', default=False)
args = parser.parse_args()

weights_path = args.weights
image_path   = args.image
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
output_path  = args.output_image
NUM_CLASSES  = args.num_classes
name_file    = args.class_names
NO_GPU       = args.nogpu

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#if NO_GPU:
#    device = torch.device("cpu")
#tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# クラスファイルからクラス名を読み込む
class_names = []
with open(name_file, 'r') as f:
    class_names = f.read().splitlines()

# ONNX形式のモデル読み込み
model = onnx.load(weights_path)

# Tensorflow形式のモデルに変換
tf_model = onnx_tf.backend.prepare(model, device='CPU')

# 画像パスから入力画像データに変換
start = time.time();

input_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

image_load_t = time.time()
input_image = input_image[:, :, :] / 255.0
image = resize_aspect_ratio(input_image, use_torch=False)      # by opencv
image = image.transpose(2, 0, 1)
image = image[np.newaxis, [2,1,0], :, :]
np.set_printoptions(edgeitems=2000)
image = tf.cast(image, tf.float32)
image_convert_t = time.time()

# 入力画像からモデルによる推論を実行する
output = tf_model.run(image)       # 出力座標は 0~1 の値

inference_t = time.time()

# 推論結果に NMS をかける
# ここの outputの出力座標 はすでに 0~416 にスケールされている
output_torch = torch.from_numpy(output[0])
output = non_max_suppression(output_torch, conf_thres, nms_thres)
nms_boxes = output[0]
print("nms_boxes :", nms_boxes.shape)

# <!> tf の nms() は, boxes が [y1, x1, y2, x2] を想定している
#     output は, [x1, y1, x2, y2, conf, cls] の順番なので注意
#boxes  = output[0][0][:, 0:4]
#boxes[:, 0] = boxes[:, 1] / 416.0     # 正規化処理
#boxes[:, 1] = boxes[:, 0] / 416.0
#boxes[:, 2] = boxes[:, 3] / 416.0
#boxes[:, 3] = boxes[:, 2] / 416.0
#boxes  = boxes[:, [1,0,3,2]]
#scores = output[0][0][:, 4]
#selected_indices = tf.image.non_max_suppression(boxes, scores, 200, iou_threshold=0.3, score_threshold=0.01)
#
nms_t = time.time()
#
##output = output[0]
#nms_boxes = tf.gather(boxes, selected_indices)
#nms_boxes = nms_boxes * 416.0

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

# 出力画像の下地として, 入力画像を読み込む
#plt.figure()
#fig, ax = plt.subplots(1)
#ax.imshow(rgb_image)

### クラスによって描く色を決める
#cmap = plt.get_cmap('tab20b')       # tab20b はカラーマップの種類の1つ
#colors = [cmap(i) for i in np.linspace(0, 1, NUM_CLASSES)]  # cmap をリスト化 (80分割)
#bbox_colors = random.sample(colors, NUM_CLASSES)     # カラーをランダムに並び替え (任意)

### 推論結果(x_min, y_min, x_max, y_max, confidence, class) をもとに
### 描画する矩形とラベルを作成する
for x_min, y_min, x_max, y_max in unpad_boxes:
    box_w = x_max - x_min
    box_h = y_max - y_min

    #color = bbox_colors[int(class_pred)]
    ## patches を使うと, 図の上に図形をかける?
    #bbox = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor=color, facecolor='None')
    #ax.add_patch(bbox)

    ## ラベル
    #plt.text(x_min, y_min, s=class_names[int(class_pred)], color='white', verticalalignment='top', bbox={'color': color, 'pad':0})
    cv2.rectangle(input_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), thickness=2)
    cv2.putText(input_image, 'doll', (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 255), thickness=2)

end = time.time()
print("elapsed time = %.4f sec" % (end - start))
print("items :")
print(" image_load : %.4f sec" % (image_load_t - start))
print(" image_convert : %.4f sec" % (image_convert_t - image_load_t))
print(" inference : %.4f sec" % (inference_t - image_convert_t))
print(" nms : %.4f sec" % (nms_t - inference_t))
print(" plot : %.4f sec" % (end - nms_t))

# 描画する
#plt.axis("off")     # 軸をオフにする
#plt.savefig(output_path)
#plt.close()
cv2.imshow('image', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
