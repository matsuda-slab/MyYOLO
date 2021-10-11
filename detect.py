from utils.transforms import Resize, DEFAULT_TRANSFORMS
from utils.transforms_detect import resize_aspect_ratio
import torch
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from model import YOLO, load_model
from utils.utils import non_max_suppression
import time

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='weights/tiny-yolo.model')
parser.add_argument('--image', default='images/dog.jpg')
parser.add_argument('--conf_thres', type=float, default=0.5)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--output_image', default='output.jpg')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--class_names', default='coco.names')
args = parser.parse_args()

weights_path = args.weights
image_path   = args.image
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
model = load_model(weights_path, device, num_classes=NUM_CLASSES)

# 画像パスから入力画像データに変換
#input_image = Image.open(image_path).convert('RGB')
#resizer     = transforms.Resize((416, 416), interpolation=2)    # nearest
#resized_image = resizer(input_image)
#resized_image = np.array(resized_image, dtype=np.uint8)
#image       = torch.from_numpy(resized_image).to(device)
#image       = image.permute(2, 0, 1)
#image       = image.unsqueeze(0)
#image       = image.type(tensor_type)
start = time.time();
#   input_image = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
#   image = transforms.Compose([
#       DEFAULT_TRANSFORMS,
#       Resize(416)])((input_image, np.zeros((1,5))))[0].unsqueeze(0)
#   image = image.to(device)
#np.set_printoptions(threshold=np.inf)
#print(image.detach().cpu().numpy())

input_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
image = transforms.ToTensor()(input_image)

image = resize_aspect_ratio(image)
image = torch.from_numpy(image)
image = image.to(device)
#image = torch.from_numpy(image).to(device)
image = image.permute(2, 0, 1)
image = image[[2,1,0],:,:]
image = image.unsqueeze(0)
#image = image.type(tensor_type)

#resized_image = resized_image.squeeze(0)
#resized_image = resized_image.transpose(1,2,0)
#resized_image = resized_image[:, :, ::-1]
#cv2.imshow('image_resized', image)
#cv2.waitKey(0)

#np.set_printoptions(threshold=np.inf)
#image_print = image.detach().cpu().numpy()
#print(image_print)

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

# 出力画像の下地として, 入力画像を読み込む
plt.figure()
fig, ax = plt.subplots(1)
ax.imshow(rgb_image)

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

end = time.time()
print("elapsed time = %.4f sec" % (end - start))

# 描画する
plt.axis("off")     # 軸をオフにする
plt.savefig(output_path)
plt.close()
