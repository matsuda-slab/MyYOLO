"""
パラメータの値の範囲のカウントをファイルに出力する.
同時に, パラメータのすべての値を, レイヤごとにファイルに出力する
"""

import os, sys
import torch
import argparse
import numpy as np

sys.path.append('../../../')
import utils.count_distribution as distrib

#sys.path.append('../../../')
from utils.utils import plot_distrib

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='weights/yolov3-tiny/yolov3-tiny.pt')
parser.add_argument('--savedir', default='yolov3-tiny')
parser.add_argument('--mode', default='torch')
parser.add_argument('--num_classes', type=int, default=80)
args = parser.parse_args()

weights_path = args.weights
NUM_CLASSES  = args.num_classes

if args.mode == 'torch':
    # load weight
    params = torch.load(weights_path, map_location='cpu')

    cnt_class_values = np.zeros(10).astype(int)

    for key in params.keys():
        param = params[key]
        if not "num_batches_tracked" in key:
            if param is None:
                continue
            distrib.count(cnt_class_values, param)

    print("[0-1) :", cnt_class_values[1])
    print("[1-2) :", cnt_class_values[2])
    print("[2-4) :", cnt_class_values[3])
    print("[4-8) :", cnt_class_values[4])
    print("[8-16) :", cnt_class_values[5])
    print("[16-32) :", cnt_class_values[6])
    print("[32-64) :", cnt_class_values[7])
    print("[64-128) :", cnt_class_values[8])
    print("[128 :", cnt_class_values[9])

    plot_distrib(cnt_class_values, savedir=args.savedir, graph_name="weight_distribution")

elif args.mode == 'numpy':
    # load weight
    params = np.load(weights_path, allow_pickle=True)
    params = params.item()

    cnt_class_values = np.zeros(10).astype(int)

    for key in params.keys():
        print(key)
        param = params[key]
        if param is None:
            continue
        distrib.count_np(cnt_class_values, param)

    print("[0-1) :", cnt_class_values[1])
    print("[1-2) :", cnt_class_values[2])
    print("[2-4) :", cnt_class_values[3])
    print("[4-8) :", cnt_class_values[4])
    print("[8-16) :", cnt_class_values[5])
    print("[16-32) :", cnt_class_values[6])
    print("[32-64) :", cnt_class_values[7])
    print("[64-128) :", cnt_class_values[8])
    print("[128 :", cnt_class_values[9])

    plot_distrib(cnt_class_values, savedir=args.savedir, graph_name="weight_distribution")

else:
    print("args error: mode is either 'torch' or 'numpy'")
    sys.exit(1)

