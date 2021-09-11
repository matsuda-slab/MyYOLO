import torch
from utils.datasets import _create_validation_data_loader
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
from model import YOLO, load_model
from utils.utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='weights/tiny-yolo.model')
parser.add_argument('--conf_thres', default=0.01)
parser.add_argument('--nms_thres', default=0.4)
parser.add_argument('--iou_thres', default=0.5)
parset.add_argument('--class_file', default='coco.names')
parser.add_argument('--data_root', default='/home/matsuda/datasets/COCO/2014')
args = parser.parse_args()

NUM_CLASSES  = 80
BATCH_SIZE   = 8
IMG_SIZE     = 416
DATA_ROOT    = args.data_root
VALID_PATH   = DATA_ROOT + '/5k.txt'

weights_path = args.weights
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
iou_thres    = args.iou_thres
class_file   = args.class_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# クラスファイルからクラス名を読み込む
class_names = []
with open(class_file, 'r') as f:
    class_names = f.read().splitlines()

# モデルファイルからモデルを読み込む
#model = YOLO()
#model.load_state_dict(torch.load(weights_path, map_location=device))
#model.to(device)
model = load_model(weights_path, device)

# valid用のデータローダを作成する
dataloader = _create_validation_data_loader(
        VALID_PATH,
        BATCH_SIZE,
        IMG_SIZE
        )

# 推論実行
model.eval()

labels         = []
sample_metrics = []
for _, images, targets in dataloader:
    # ラベル(番号)をリスト化している (あとで必要なのだろう)
    labels += targets[:, 1].tolist()

    images = images.type(tensor_type)

    # w, h を x, y に直すのは, あとの関数で必要なのだろう
    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    targets[:, 2:] *= IMG_SIZE

    with torch.no_grad():
        outputs = model(images)
    # nmsをかける
        outputs = non_max_suppression(outputs, conf_thres, nms_thres)

# スコア(precision, recall, TPなど)を算出する
    sample_metrics += get_batch_statistics(outputs, targets, iou_thres)

# クラスごとの AP を算出する
TP, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
metrics_output = ap_per_class(TP, pred_scores, pred_labels, labels)

# mAP を算出する
precision, recall, AP, f1, ap_class = metrics_output
ap_table = [['Index', 'Class', 'AP']]
for i, c in enumerate(ap_class):
    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
for ap in ap_table:
    print(ap)

mAP = AP.mean() 
print("mAP :", mAP)
