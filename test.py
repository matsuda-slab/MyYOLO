import torch
from utils.datasets import _create_validation_data_loader
import os
import sys
import argparse
import numpy as np
import random
from model import load_model
from utils.utils import non_max_suppression, xywh2xyxy,
                        get_batch_statistics, ap_per_class
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='weights/yolov3-tiny.weights')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--conf_thres', type=float, default=0.01)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--iou_thres', type=float, default=0.5)
parser.add_argument('--class_names', default='namefiles/coco.names')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--data_root', default='/home/matsuda/datasets/COCO/2014')
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--nogpu', action='store_true', default=False)
parser.add_argument('--notiny', action='store_true', default=False)
args = parser.parse_args()

IMG_SIZE     = 416
DATA_ROOT    = args.data_root
VALID_PATH   = DATA_ROOT + '/5k.txt' 
                if 'COCO' in DATA_ROOT 
                else DATA_ROOT + '/test.txt'
BATCH_SIZE   = args.batch_size
weights_path = args.weights
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
iou_thres    = args.iou_thres
class_file   = args.class_names
NUM_CLASSES  = args.num_classes
EN_TINY      = not args.notiny

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.nogpu:
    device = torch.device("cpu")
tensor_type = torch.cuda.FloatTensor 
                if torch.cuda.is_available() 
                else torch.FloatTensor
if args.nogpu:
    tensor_type = torch.FloatTensor

# クラスファイルからクラス名を読み込む
class_names = []
with open(class_file, 'r') as f:
    class_names = f.read().splitlines()

# モデルファイルからモデルを読み込む
model = load_model(weights_path, device, tiny=EN_TINY, num_classes=NUM_CLASSES,
                    quant=args.quant)

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
for _, images, targets in tqdm.tqdm(dataloader):
    # ラベル(番号)をリスト化している (あとで必要なのだろう)
    labels += targets[:, 1].tolist()

    # w, h を x, y に直すのは, あとの関数で必要なのだろう
    targets[:, 2:] = xywh2xyxy(targets[:, 2:])
    targets[:, 2:] *= IMG_SIZE

    images = images.type(tensor_type)

    with torch.no_grad():
        outputs = model(images)
        #""" debug """
        #with open("debug.txt", "a") as df:
        #    df.write(str(outputs[0, :, 2]))
        #"""       """
    # nmsをかける
        outputs = non_max_suppression(outputs, conf_thres, nms_thres)

# スコア(precision, recall, TPなど)を算出する
    sample_metrics += get_batch_statistics(outputs, targets, iou_thres)

# クラスごとの AP を算出する
TP, pred_scores, pred_labels 
    = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
metrics_output = ap_per_class(TP, pred_scores, pred_labels, labels)

# mAP を算出する
precision, recall, AP, f1, ap_class = metrics_output
ap_table = [['Index', 'Class', 'AP']]
for i, c in enumerate(ap_class):
    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
for ap in ap_table:
    print(ap)

mAP = AP.mean() 
print("mAP : %.5f" % mAP)
