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
from utils import non_max_suppression

NUM_CLASSES = 80

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='tiny-yolo.model')
parser.add_argument('--conf_thres', default=0.5)
parser.add_argument('--nms_thres', default=0.4)
args = parser.parse_args()

weights_path = args.weights
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres

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

# valid用のデータローダを作成する

# 推論実行

# nmsをかける

# スコア(precision, recall, TPなど)を算出する

# クラスごとの AP を算出する

# mAP を算出する
