#===============================================================================
# Pytorch の Quantize API を使って, YOLO のモデルを 8bit-Int に量子化する
#===============================================================================

import os
import torch
from model import load_model
import argparse
from torchvision import transforms
from utils.datasets import _create_validation_data_loader
import cv2
from utils.utils import non_max_suppression, xywh2xyxy, get_batch_statistics, ap_per_class
from utils.transforms_detect import resize_aspect_ratio
import numpy as np

CONF_THRES = 0.1
NMS_THRES  = 0.4

def detect(model, device, tensor_type, data_path, batch_size=8, class_file='coco.names'):
    img_size    = 416
    conf_thres  = 0.1
    nms_thres   = 0.4
    iou_thres   = 0.5
    #input_image = cv2.imread(image_path)
    #rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    #image = transforms.ToTensor()(input_image)      # ここで 0~1のfloatになる

    #image = resize_aspect_ratio(image)
    #image = torch.from_numpy(image)
    #image = image.to(device)
    #image = image.permute(2, 0, 1)
    #image = image[[2,1,0],:,:]
    #image = image.unsqueeze(0)
    class_names = []
    with open(class_file, 'r') as f:
        class_names = f.read().splitlines()

    dataloader = _create_validation_data_loader(
            data_path,
            batch_size,
            img_size
            )

    # 入力画像からモデルによる推論を実行する
    model.eval()

    labels         = []
    sample_metrics = []
    for _, images, targets in dataloader:
        # ラベル(番号)をリスト化している (あとで必要なのだろう)
        labels += targets[:, 1].tolist()

        # w, h を x, y に直すのは, あとの関数で必要なのだろう
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        images = images.type(tensor_type)

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
    print("mAP : %.5f" % mAP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='yolo-tiny.pt')
    parser.add_argument('--data_root', default='/home/matsuda/datasets/contest_doll_light')
    parser.add_argument('--num_classes', type=int, default=80)
    args = parser.parse_args()

    weights_path = args.weights
    NUM_CLASSES  = args.num_classes

    device       = torch.device('cpu')

    qmodel = load_model(weights_path, device, num_classes=NUM_CLASSES, quant=True, qconvert=False)
    qmodel.eval()

    qmodel.qconfig = torch.quantization.default_qconfig
    print("qconfig :", qmodel.qconfig)
    torch.quantization.prepare(qmodel, inplace=True)

    # 適当な入力画像
    #image_path = 'images/doll_light_1.png'
    testdata_path = args.data_root + '/5k.txt' if 'COCO' in args.data_root else args.data_root + '/test.txt'

    # forward を回して, 量子化に必要な scale と zero-point を決定する
    #detect(qmodel, device, image_path)
    detect(qmodel, device, torch.FloatTensor, testdata_path, class_file='contest_2.names')

    #print("param:", qmodel.state_dict()['conv1.conv.weight'])
    torch.quantization.convert(qmodel, inplace=True)
    #print("param:", qmodel.state_dict()['conv1.conv.weight'])

    print("qmodel :", qmodel)
    detect(qmodel, device, torch.FloatTensor, testdata_path, class_file='contest_2.names')

    inputfilename, ext = os.path.splitext(weights_path)
    outputfilename = inputfilename + "_quant" + ext
    torch.save(qmodel.state_dict(), outputfilename)

if __name__ == "__main__":
    main()
