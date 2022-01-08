"""
パラメータファイルをロードし, すべての値をファイルに出力する
量子化前と量子化後での値の比較に使う
"""

import sys, os
import argparse

import torch
import torch.nn as nn

from model import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='yolo-tiny_doll.pt')
parser.add_argument('--num_classes', type=int, default=1)
args = parser.parse_args()

weights_path = args.weights
NUM_CLASSES  = args.num_classes
quant        = True if 'quant' in weights_path else False

# load model
device = torch.device("cpu")
model  = load_model(weights_path, device, num_classes=NUM_CLASSES, quant=quant)

keys   = str(model.state_dict().keys())
print(keys.replace(',', '\n'))

for key in model.state_dict().keys():
    a = model.state_dict()[key]
    print(key)
    if a is None:
        continue
    if 'quant' in weights_path:
        output_path = os.path.join('debug_params', 'quant', str(key) + '.txt')
    else:
        output_path = os.path.join('debug_params', 'no-quant', str(key) + '.txt')
    with open(output_path, "w") as f:
        if (0 < a.dim()):
            for aa in a:
                if (0 < aa.dim()):                              # conv.weight
                    for aaa in aa:
                        for aaaa in aaa:
                            for aaaaa in aaaa:
                                if quant:
                                    aaaaa = aaaaa.dequantize()
                                f.write(str(aaaaa.detach().numpy()) + '\n')
                else:
                    f.write(str(aa.detach().numpy()) + '\n')                  # conv.bias, bn
        else:                                                   # bn.bias, bn.nbt
            f.write(str(a.detach().numpy()) + '\n')
