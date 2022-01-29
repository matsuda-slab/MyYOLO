from utils.transforms_detect import resize_aspect_ratio
import torch
import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from torchvision import transforms
import cv2
from model import load_model
from utils.utils import non_max_suppression, plot_distrib
import time
#matplotlib.use('TkAgg')

def extract(target, inputs):
    feature = None

    def forward_hook(module, inputs, outputs):
        global features
        features = outputs.detach().clone()

    handle = target.register_forward_hook(forward_hook)

    model.eval()
    model(inputs, distri_array=activate_distrib, debug=args.debug)

    handle.remove()

    return features

parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None)
parser.add_argument('--weights', default='weights/yolov3-tiny.pt')
parser.add_argument('--image', default='images/car29.jpg')
parser.add_argument('--conf_thres', type=float, default=0.1)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--output_image', default='output.jpg')
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--class_names', default='namefiles/car.names')
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--nogpu', action='store_true', default=False)
parser.add_argument('--notiny', action='store_true', default=False)
parser.add_argument('--distrib', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--merge', action='store_true', default=False)
args = parser.parse_args()

weights_path = args.weights
image_path   = args.image
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
output_path  = args.output_image
NUM_CLASSES  = args.num_classes
name_file    = args.class_names
NO_GPU       = args.nogpu
EN_TINY      = not args.notiny
SEP          = True if args.model == "sep" else False
MERGE        = args.merge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NO_GPU or args.quant:
    device = torch.device("cpu")
tensor_type = (torch.cuda.FloatTensor
                if (torch.cuda.is_available() and not NO_GPU)
                else torch.FloatTensor)
if args.quant:
    tensor_type = torch.ByteTensor

# クラスファイルからクラス名を読み込む
class_names = []
with open(name_file, 'r') as f:
    class_names = f.read().splitlines()

# モデルファイルからモデルを読み込む
model = load_model(weights_path, device, merge=MERGE, tiny=EN_TINY,
                   num_classes=NUM_CLASSES, quant=args.quant, jit=True,
                   use_sep=SEP)

# 画像パスから入力画像データに変換
start = time.time();

input_image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
image = transforms.ToTensor()(input_image)      # ここで 0~1のfloatになる

image_load_t = time.time()

image = resize_aspect_ratio(image)
image = torch.from_numpy(image)
image = image.to(device)
image = image.permute(2, 0, 1)
image = image[[2,1,0],:,:]
image = image.unsqueeze(0)

image_convert_t = time.time()

# 入力画像からモデルによる推論を実行する
model.eval()

activate_distrib = np.zeros(10)
output = model(image, distri_array=activate_distrib, debug=args.debug)       # 出力座標は 0~1 の値

l1_conv_dw_output = extract(model.conv1.conv_dw, image)

if args.distrib:
    print("all :", activate_distrib[0])
    print("distrib :", "%.2f" % ((activate_distrib[1] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[2] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[3] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[4] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[5] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[6] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[7] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[8] / activate_distrib[0]) * 100), "%")
    print("distrib :", "%.2f" % ((activate_distrib[8] / activate_distrib[0]) * 100), "%")

    plot_distrib(activate_distrib)

inference_t = time.time()

# 推論結果に NMS をかける
# ここの outputの出力座標 はすでに 0~416 にスケールされている
output = non_max_suppression(output, conf_thres, nms_thres)

nms_t = time.time()

output = output[0]
#print(output)
print("output :", output.shape)

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
# cmap をリスト化 (80分割)
colors = [cmap(i) for i in np.linspace(0, 1, NUM_CLASSES)]
# カラーをランダムに並び替え (任意)
bbox_colors = random.sample(colors, NUM_CLASSES)

### 推論結果(x_min, y_min, x_max, y_max, confidence, class) をもとに
### 描画する矩形とラベルを作成する
for x_min, y_min, x_max, y_max, conf, class_pred in output:
    box_w = x_max - x_min
    box_h = y_max - y_min

    color = bbox_colors[int(class_pred)]
    bbox = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2,
                        edgecolor=color, facecolor='None')
    ax.add_patch(bbox)

    # ラベル
    plt.text(x_min, y_min, s=class_names[int(class_pred)], color='white',
                        verticalalignment='top', bbox={'color': color, 'pad':0})

end = time.time()
print("elapsed time = %.4f sec" % (end - start))
print("items :")
print(" image_load : %.4f sec" % (image_load_t - start))
print(" image_convert : %.4f sec" % (image_convert_t - image_load_t))
print(" inference : %.4f sec" % (inference_t - image_convert_t))
print(" nms : %.4f sec" % (nms_t - inference_t))
print(" plot : %.4f sec" % (end - nms_t))

# 描画する
plt.axis("off")
plt.savefig(output_path)
plt.show()
plt.close()
