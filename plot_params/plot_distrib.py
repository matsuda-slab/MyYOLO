import os, sys
import torch
import argparse
import numpy as np

sys.path.append('../')

from model import load_model

def count(array, param):
    param_abs = abs(param)
    if param_abs < 1:
        array[0] = array[0] + 1
    elif param_abs < 2:
        array[1] = array[1] + 1
    elif param_abs < 4:
        array[2] = array[2] + 1
    elif param_abs < 8:
        array[3] = array[3] + 1
    elif param_abs < 16:
        array[4] = array[4] + 1
    elif param_abs < 32:
        array[5] = array[5] + 1
    elif param_abs < 64:
        array[6] = array[6] + 1
    elif param_abs < 128:
        array[7] = array[7] + 1
    else:
        print(param)
        array[8] = array[8] + 1

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

cnt_class_values = np.zeros(9).astype(int)

for key in model.state_dict().keys():
    a = model.state_dict()[key]
    print(key)
    if a is None:
        continue
    output_path = os.path.join('params', str(key) + '.txt')
    with open(output_path, "w") as f:
        if (0 < a.dim()):
            for aa in a:
                if (0 < aa.dim()):                              # conv.weight
                    for aaa in aa:
                        for aaaa in aaa:
                            for aaaaa in aaaa:
                                if quant:
                                    aaaaa = aaaaa.dequantize()
                                param = aaaaa.detach().numpy()
                                f.write(str(param) + '\n')
                                count(cnt_class_values, param)
                else:
                    param = aa.detach().numpy()
                    f.write(str(param) + '\n')                  # conv.bias, bn
                    count(cnt_class_values, param)
        else:                                                   # bn.bias, bn.nbt
            param = a.detach().numpy()
            f.write(str(a.detach().numpy()) + '\n')
            count(cnt_class_values, param)

with open('params_class_count.txt', "w") as f:
    f.write('[0, 1) : ' + str(cnt_class_values[0]) + '\n')
    f.write('[1, 2) : ' + str(cnt_class_values[1]) + '\n')
    f.write('[2, 4) : ' + str(cnt_class_values[2]) + '\n')
    f.write('[4, 8) : ' + str(cnt_class_values[3]) + '\n')
    f.write('[8, 16) : ' + str(cnt_class_values[4]) + '\n')
    f.write('[16, 32) : ' + str(cnt_class_values[5]) + '\n')
    f.write('[32, 64) : ' + str(cnt_class_values[6]) + '\n')
    f.write('[64, 128) : ' + str(cnt_class_values[7]) + '\n')
    f.write('[128, : ' + str(cnt_class_values[8]))
