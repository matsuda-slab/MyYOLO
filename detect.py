import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import YOLO
from utils.utils import non_max_suppression

NUM_CLASSES = 80

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='tiny-yolo.model')
parser.add_argument('--image', default='images/dog.jpg')
parser.add_argument('--conf_thres', default=0.5)
parser.add_argument('--nms_thres', default=0.4)
parser.add_argument('--output_image', default='output.jpg')
args = parser.parse_args()

weights_path = args.weights
image_path   = args.image
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
output_path  = args.output_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# クラスファイルからクラス名を読み込む
class_names = []
with open('coco.names', 'r') as f:
    class_names = f.read().splitlines()

# モデルファイルからモデルを読み込む
model = YOLO()
model.load_state_dict(torch.load(weights_path, map_location=device))
model.to(device)

# 画像パスから入力画像データに変換
input_image = Image.open(image_path).convert('RGB')
input_image = np.array(input_image, dtype=np.uint8)
image       = torch.from_numpy(input_image).to(device)
image       = image.permute(2, 0, 1)
resizer     = transforms.Resize((416, 416), interpolation=2)    # nearest
image       = resizer(image).unsqueeze(0)
image       = image.type(tensor_type)

# 入力画像からモデルによる推論を実行する
model.eval()
output = model(image)       # 出力座標は 0~1 の値

# 推論結果に NMS をかける
# ここの outputの出力座標 はすでに 0~416 にスケールされている
output = non_max_suppression(output, conf_thres, nms_thres)

output = output[0]

### 推論結果のボックスの位置(0~1)を元画像のサイズに合わせてスケールする
orig_h, orig_w = input_image.shape[:2]
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

# 出力画像の下地として, 入力画像を読み込む
plt.figure()
fig, ax = plt.subplots(1)
ax.imshow(input_image)

### クラスによって描く色を決める
cmap = plt.get_cmap('tab20b')       # tab20b はカラーマップの種類の1つ
colors = [cmap(i) for i in np.linspace(0, 1, NUM_CLASSES)]  # cmap をリスト化 (80分割)
bbox_colors = random.sample(colors, NUM_CLASSES)     # カラーをランダムに並び替え (任意)

### 推論結果(x_min, y_min, x_max, y_max, confidence, class) をもとに
### 描画する矩形とラベルを作成する
for x_min, y_min, x_max, y_max, conf, class_pred in output:
    box_w = x_max - x_min
    box_h = y_max - y_min

    color = bbox_colors[int(class_pred)]
    # patches を使うと, 図の上に図形をかける?
    bbox = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2, edgecolor=color, facecolor='None')
    ax.add_patch(bbox)

    # ラベル
    plt.text(x_min, y_min, s=class_names[int(class_pred)], color='white', verticalalignment='top', bbox={'color': color, 'pad':0})

# 描画する
plt.axis("off")     # 軸をオフにする
plt.savefig(output_path)
plt.close()
