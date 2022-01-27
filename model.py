import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from collections import OrderedDict
import numpy as np
import time
import utils.count_distribution as ditsrib

# YOLOv3-tiny
class YOLO_tiny(nn.Module):
    def __init__(self, num_classes, quant=False, dropout=False):
        super(YOLO_tiny, self).__init__()
        #self.anchors     = [[[10,14], [23,27], [37,58]], [[81,82], [135,169], [344,319]]]
        self.anchors     = [[[23,27], [37,58], [81,82]], [[81,82], [135,169], [344,319]]]
        self.img_size    = 416
        self.num_classes = num_classes
        self.ylch        = (5 + self.num_classes) * 3       # yolo layer channels
        #self.p_dropout   = p_dropout

        # modules
        self.conv1 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d(   3,   16, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d(  16, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv2 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d(  16,   32, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d(  32, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv3 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d(  32,   64, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d(  64, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv4 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d(  64,  128, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d( 128, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv5 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d( 128,  256, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv6 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d( 256,  512, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d( 512, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv7 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d( 512, 1024, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d(1024, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv8 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d(1024,  256, kernel_size=1,
                                         stride=1, padding=0, bias=0)),
                         ('bn',   nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv9 = nn.Sequential(OrderedDict([
                         ('conv', nn.Conv2d( 256,  512, kernel_size=3,
                                         stride=1, padding=1, bias=0)),
                         ('bn',   nn.BatchNorm2d( 512, momentum=0.1, eps=1e-5)),
                         ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv10 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d( 512, self.ylch, kernel_size=1,
                                          stride=1, padding=0, bias=1))
                      ]))
        self.conv11 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d( 256,  128, kernel_size=1,
                                          stride=1, padding=0, bias=0)),
                          ('bn',   nn.BatchNorm2d( 128, momentum=0.1, eps=1e-5)),
                          ('relu', nn.LeakyReLU(0.1))
                      ]))
        self.conv12 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d( 384,  256, kernel_size=3,
                                          stride=1, padding=1, bias=0)),
                          ('bn',   nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                          ('relu', nn.LeakyReLU(0.1))
                      ]))
        self.conv13 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv2d( 256, self.ylch, kernel_size=1,
                                          stride=1, padding=0, bias=1))
                      ]))
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool5  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool6  = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.zeropad = nn.ZeroPad2d((0, 1, 0, 1)) # サイズを保つための
                                                   # ゼロパディング (pool6の直前)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.yolo1    = YOLOLayer(self.anchors[1],
                            self.img_size, self.num_classes)
        self.yolo2    = YOLOLayer(self.anchors[0],
                            self.img_size, self.num_classes)

        self.en_dropout = dropout
        self.dropout  = nn.Dropout2d(p=0.5)

        self.yolo_layers = [self.yolo1, self.yolo2]

        # 量子化器
        self.enquant  = quant
        self.quant    = QuantStub()
        self.dequant  = DeQuantStub()

    def load_weights(self, weights_path, device):
        ckpt = torch.load(weights_path, map_location=device)
        param = ckpt['model']

        self.conv1.weight = nn.Parameter(
                                    param['module_list.0.Conv2d.weight'])
        self.bn1.weight   = nn.Parameter(
                                    param['module_list.0.BatchNorm2d.weight'])
        self.bn1.bias     = nn.Parameter(
                                    param['module_list.0.BatchNorm2d.bias'])
        self.bn1.running_mean = param['module_list.0.BatchNorm2d.running_mean']
        self.bn1.running_var  = param['module_list.0.BatchNorm2d.running_var']
        self.bn1.num_batches_tracked \
            = param['module_list.0.BatchNorm2d.num_batches_tracked']
        self.conv2.weight \
            = nn.Parameter(param['module_list.2.Conv2d.weight'])
        self.bn2.weight \
            = nn.Parameter(param['module_list.2.BatchNorm2d.weight'])
        self.bn2.bias \
            = nn.Parameter(param['module_list.2.BatchNorm2d.bias'])
        self.bn2.running_mean = param['module_list.2.BatchNorm2d.running_mean']
        self.bn2.running_var  = param['module_list.2.BatchNorm2d.running_var']
        self.bn2.num_batches_tracked \
            = param['module_list.2.BatchNorm2d.num_batches_tracked']
        self.conv3.weight \
            = nn.Parameter(param['module_list.4.Conv2d.weight'])
        self.bn3.weight \
            = nn.Parameter(param['module_list.4.BatchNorm2d.weight'])
        self.bn3.bias \
            = nn.Parameter(param['module_list.4.BatchNorm2d.bias'])
        self.bn3.running_mean = param['module_list.4.BatchNorm2d.running_mean']
        self.bn3.running_var  = param['module_list.4.BatchNorm2d.running_var']
        self.bn3.num_batches_tracked \
            = param['module_list.4.BatchNorm2d.num_batches_tracked']
        self.conv4.weight \
            = nn.Parameter(param['module_list.6.Conv2d.weight'])
        self.bn4.weight \
            = nn.Parameter(param['module_list.6.BatchNorm2d.weight'])
        self.bn4.bias \
            = nn.Parameter(param['module_list.6.BatchNorm2d.bias'])
        self.bn4.running_mean = param['module_list.6.BatchNorm2d.running_mean']
        self.bn4.running_var  = param['module_list.6.BatchNorm2d.running_var']
        self.bn4.num_batches_tracked \
            = param['module_list.6.BatchNorm2d.num_batches_tracked']
        self.conv5.weight \
            = nn.Parameter(param['module_list.8.Conv2d.weight'])
        self.bn5.weight \
            = nn.Parameter(param['module_list.8.BatchNorm2d.weight'])
        self.bn5.bias \
            = nn.Parameter(param['module_list.8.BatchNorm2d.bias'])
        self.bn5.running_mean = param['module_list.8.BatchNorm2d.running_mean']
        self.bn5.running_var  = param['module_list.8.BatchNorm2d.running_var']
        self.bn5.num_batches_tracked \
            = param['module_list.8.BatchNorm2d.num_batches_tracked']
        self.conv6.weight \
            = nn.Parameter(param['module_list.10.Conv2d.weight'])
        self.bn6.weight \
            = nn.Parameter(param['module_list.10.BatchNorm2d.weight'])
        self.bn6.bias \
            = nn.Parameter(param['module_list.10.BatchNorm2d.bias'])
        self.bn6.running_mean = param['module_list.10.BatchNorm2d.running_mean']
        self.bn6.running_var  = param['module_list.10.BatchNorm2d.running_var']
        self.bn6.num_batches_tracked \
            = param['module_list.10.BatchNorm2d.num_batches_tracked']

        self.conv7.weight \
            = nn.Parameter(param['module_list.12.Conv2d.weight'])
        self.bn7.weight \
            = nn.Parameter(param['module_list.12.BatchNorm2d.weight'])
        self.bn7.bias \
            = nn.Parameter(param['module_list.12.BatchNorm2d.bias'])
        self.bn7.running_mean = param['module_list.12.BatchNorm2d.running_mean']
        self.bn7.running_var  = param['module_list.12.BatchNorm2d.running_var']
        self.bn7.num_batches_tracked \
            = param['module_list.12.BatchNorm2d.num_batches_tracked']
        self.conv8.weight \
            = nn.Parameter(param['module_list.13.Conv2d.weight'])
        self.bn8.weight \
            = nn.Parameter(param['module_list.13.BatchNorm2d.weight'])
        self.bn8.bias \
            = nn.Parameter(param['module_list.13.BatchNorm2d.bias'])
        self.bn8.running_mean = param['module_list.13.BatchNorm2d.running_mean']
        self.bn8.running_var  = param['module_list.13.BatchNorm2d.running_var']
        self.bn8.num_batches_tracked \
            = param['module_list.13.BatchNorm2d.num_batches_tracked']
        self.conv9.weight \
            = nn.Parameter(param['module_list.14.Conv2d.weight'])
        self.bn9.weight \
            = nn.Parameter(param['module_list.14.BatchNorm2d.weight'])
        self.bn9.bias \
            = nn.Parameter(param['module_list.14.BatchNorm2d.bias'])
        self.bn9.running_mean = param['module_list.14.BatchNorm2d.running_mean']
        self.bn9.running_var  = param['module_list.14.BatchNorm2d.running_var']
        self.bn9.num_batches_tracked \
            = param['module_list.14.BatchNorm2d.num_batches_tracked']

        self.conv10.weight \
            = nn.Parameter(param['module_list.15.Conv2d.weight'])
        self.conv10.bias      = nn.Parameter(param['module_list.15.Conv2d.bias'])

        self.conv11.weight \
            = nn.Parameter(param['module_list.18.Conv2d.weight'])
        self.bn10.weight \
            = nn.Parameter(param['module_list.18.BatchNorm2d.weight'])
        self.bn10.bias \
            = nn.Parameter(param['module_list.18.BatchNorm2d.bias'])
        self.bn10.running_mean \
            = param['module_list.18.BatchNorm2d.running_mean']
        self.bn10.running_var  = param['module_list.18.BatchNorm2d.running_var']
        self.bn10.num_batches_tracked \
            = param['module_list.18.BatchNorm2d.num_batches_tracked']
        self.conv12.weight \
            = nn.Parameter(param['module_list.21.Conv2d.weight'])
        self.bn11.weight \
            = nn.Parameter(param['module_list.21.BatchNorm2d.weight'])
        self.bn11.bias \
            = nn.Parameter(param['module_list.21.BatchNorm2d.bias'])
        self.bn11.running_mean \
            = param['module_list.21.BatchNorm2d.running_mean']
        self.bn11.running_var  = param['module_list.21.BatchNorm2d.running_var']
        self.bn11.num_batches_tracked \
            = param['module_list.21.BatchNorm2d.num_batches_tracked']

        self.conv13.weight \
            = nn.Parameter(param['module_list.22.Conv2d.weight'])
        self.conv13.bias      = nn.Parameter(param['module_list.22.Conv2d.bias'])

    def set_bn_params(self, layer, params, ptr):
        # bias
        num_b = layer.bias.numel()
        bn_b  = torch.from_numpy(params[ptr : ptr + num_b]).view_as(layer.bias)
        layer.bias.data.copy_(bn_b)
        ptr  += num_b
    
        # weight
        num_w = layer.weight.numel()
        bn_w  = torch.from_numpy(params[ptr : ptr + num_w]).view_as(layer.weight)
        layer.weight.data.copy_(bn_w)
        ptr  += num_w
    
        # running mean
        num_rm = layer.running_mean.numel()
        bn_rm  = torch.from_numpy(
                        params[ptr : ptr + num_rm]).view_as(layer.running_mean)
        layer.running_mean.copy_(bn_rm)
        ptr   += num_rm
    
        # running var
        num_rv = layer.running_var.numel()
        bn_rv  = torch.from_numpy(
                        params[ptr : ptr + num_rv]).view_as(layer.running_var)
        layer.running_var.copy_(bn_rv)
        ptr   += num_rv

        return ptr

    def set_conv_weights(self, layer, params, ptr):
        num_w  = layer.weight.numel()
        conv_w = torch.from_numpy(
                        params[ptr : ptr + num_w]).view_as(layer.weight)
        layer.weight.data.copy_(conv_w)
        ptr   += num_w

        return ptr
    
    def set_conv_biases(self, layer, params, ptr):
        num_b  = layer.bias.numel()
        conv_b = torch.from_numpy(
                        params[ptr : ptr + num_b]).view_as(layer.bias)
        layer.bias.data.copy_(conv_b)
        ptr   += num_b

        return ptr
    
    def load_darknet_weights(self, weights_path):
        # バイナリファイルを読み込み, 配列にデータを格納
        with open(weights_path, "rb") as f:
            weights = np.fromfile(f, dtype=np.float32)
    
        ptr = 5        # 0~4 は, ヘッダのようなものが入っている
    
        ptr = self.set_bn_params(self.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv3.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv3.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv4.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv4.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv5.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv5.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv6.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv6.conv, weights, ptr)
    
        ptr = self.set_bn_params(self.conv7.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv7.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv8.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv8.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv9.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv9.conv, weights, ptr)
    
        ptr = self.set_conv_biases(self.conv10.conv, weights, ptr)
        ptr = self.set_conv_weights(self.conv10.conv, weights, ptr)
    
        ptr = self.set_bn_params(self.conv11.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv11.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv12.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv12.conv, weights, ptr)
    
        ptr = self.set_conv_biases(self.conv13.conv, weights, ptr)
        ptr = self.set_conv_weights(self.conv13.conv, weights, ptr)

    def forward(self, x, distri_array=None, debug=False):
        yolo_outputs = []

        #step1_t = time.time()
        # 特徴抽出部
        if self.enquant: x = self.quant(x)
        x = self.conv1(x)
        if distri_array is not None: ditsrib.count(distri_array, x)
        #if debug: print("conv1 output :", x)
        if debug: print("conv1 max :", torch.max(x))
        if debug: print("conv1 min :", torch.min(x))
        x = self.pool1(x)
        if self.en_dropout: x = self.dropout(x)

        x = self.conv2(x)
        #if debug: print("conv2 output :", x)
        if debug: print("conv2 max :", torch.max(x))
        if debug: print("conv2 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.pool2(x)
        if self.en_dropout: x = self.dropout(x)

        x = self.conv3(x)
        #if debug: print("conv3 output :", x)
        if debug: print("conv3 max :", torch.max(x))
        if debug: print("conv3 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.pool3(x)
        if self.en_dropout: x = self.dropout(x)

        x = self.conv4(x)
        #if debug: print("conv4 output :", x)
        if debug: print("conv4 max :", torch.max(x))
        if debug: print("conv4 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.pool4(x)
        if self.en_dropout: x = self.dropout(x)

        x = self.conv5(x)
        #if debug: print("conv5 output :", x)
        if debug: print("conv5 max :", torch.max(x))
        if debug: print("conv5 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        l8_output = x       # あとのconcat用に出力を保管
        x = self.pool5(x)
        if self.en_dropout: x = self.dropout(x)

        x = self.conv6(x)
        #if debug: print("conv6 output :", x)
        if debug: print("conv6 max :", torch.max(x))
        if debug: print("conv6 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        if self.enquant:
            x = self.dequant(x)
        x = self.zeropad(x)
        if self.enquant:
            x = self.quant(x)
        x = self.pool6(x)
        if self.en_dropout: x = self.dropout(x)

        #step2_t = time.time()
        # スケール大 検出部
        x = self.conv7(x)
        #if debug: print("conv7 output :", x)
        if debug: print("conv7 max :", torch.max(x))
        if debug: print("conv7 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        if self.en_dropout: x = self.dropout(x)

        x1 = self.conv8(x)
        #if debug: print("conv8 output :", x1)
        if debug: print("conv8 max :", torch.max(x1))
        if debug: print("conv8 min :", torch.min(x1))
        if distri_array is not None: ditsrib.count(distri_array, x1)
        if self.en_dropout: x1 = self.dropout(x1)
        x2 = x1

        x1 = self.conv9(x1)
        #if debug: print("conv9 output :", x1)
        if debug: print("conv9 max :", torch.max(x1))
        if debug: print("conv9 min :", torch.min(x1))
        if distri_array is not None: ditsrib.count(distri_array, x1)
        if self.en_dropout: x1 = self.dropout(x1)

        x1 = self.conv10(x1)
        #if debug: print("conv10 output :", x1)
        if debug: print("conv10 max :", torch.max(x1))
        if debug: print("conv10 min :", torch.min(x1))
        if distri_array is not None: ditsrib.count(distri_array, x1)
        if self.enquant:
            x1 = self.dequant(x1)
        x1 = self.yolo1(x1)
        yolo_outputs.append(x1)

        #step3_t = time.time()
        # スケール中 検出部
        x2 = self.conv11(x2)
        if distri_array is not None: ditsrib.count(distri_array, x2)
        if self.en_dropout: x2 = self.dropout(x2)

        x2 = self.upsample(x2)
        # チャネル数方向に特徴マップを結合
        x2 = torch.cat([x2, l8_output], dim=1)

        x2 = self.conv12(x2)
        if distri_array is not None: ditsrib.count(distri_array, x2)
        if self.en_dropout: x2 = self.dropout(x2)

        x2 = self.conv13(x2)
        if distri_array is not None: ditsrib.count(distri_array, x2)
        if self.enquant:
            x2 = self.dequant(x2)
        x2 = self.yolo2(x2)
        yolo_outputs.append(x2)

        #print("yolo_outputs.shape :", yolo_outputs[0].shape)
        #print("yolo_outputs.shape :", yolo_outputs[1].shape)
        #print("torch.cat.shape :", torch.cat(yolo_outputs, 1).shape)
        #last_t = time.time()

        #if not self.training:
        #    print("step1 : %.4f, step2 : %.4f, step3 : %.4f" % 
        #            (step2_t - step1_t, step3_t - step2_t, last_t - step3_t))
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

class YOLO(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLO, self).__init__()
        self.anchors     = [[[10,13], [16,30], [33,23]],
                            [[30,61], [62,45], [59,119]],
                            [[116,90], [156,198], [373,326]]]
        self.img_size    = 416
        self.num_classes = num_classes
        self.ylch        = (5 + self.num_classes) * 3

        # modules
        self.conv1 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(3, 32, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(32, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv2 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(32, 64, kernel_size=3,
                                            stride=2, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(64, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.res1  = ResBlock(64)
        self.conv3 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(64, 128, kernel_size=3,
                                            stride=2, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(128, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.res2  = ResBlock(128)
        self.res3  = ResBlock(128)
        self.conv4 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(128, 256, kernel_size=3,
                                            stride=2, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.res4  = ResBlock(256)
        self.res5  = ResBlock(256)
        self.res6  = ResBlock(256)
        self.res7  = ResBlock(256)
        self.res8  = ResBlock(256)
        self.res9  = ResBlock(256)
        self.res10  = ResBlock(256)
        self.res11  = ResBlock(256)
        self.conv5 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, 512, kernel_size=3,
                                            stride=2, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(512, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.res12  = ResBlock(512)
        self.res13  = ResBlock(512)
        self.res14  = ResBlock(512)
        self.res15  = ResBlock(512)
        self.res16  = ResBlock(512)
        self.res17  = ResBlock(512)
        self.res18  = ResBlock(512)
        self.res19  = ResBlock(512)
        self.conv6 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, 1024, kernel_size=3,
                                            stride=2, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(1024, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.res20  = ResBlock(1024)
        self.res21  = ResBlock(1024)
        self.res22  = ResBlock(1024)
        self.res23  = ResBlock(1024)

        self.conv7 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(1024, 512, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(512, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv8 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, 1024, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(1024, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv9 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(1024, 512, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(512, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv10 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, 1024, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(1024, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv11 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(1024, 512, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(512, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv12 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, 1024, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(1024, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv13 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(1024, self.ylch, kernel_size=1,
                                            stride=1, padding=0, bias=1)),
                     ]))
        self.yolo1 = YOLOLayer(self.anchors[2], self.img_size, self.num_classes)
        self.conv14 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, 256, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv15 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(768, 256, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv16 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, 512, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(512, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv17 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, 256, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv18 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, 512, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(512, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv19 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, 256, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv20 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, 512, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(512, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv21 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(512, self.ylch, kernel_size=1,
                                            stride=1, padding=0, bias=1)),
                     ]))
        self.yolo2 = YOLOLayer(self.anchors[1], self.img_size, self.num_classes)
        self.conv22 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, 128, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(128, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv23 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(384, 128, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(128, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv24 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(128, 256, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv25 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, 128, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(128, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv26 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(128, 256, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv27 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, 128, kernel_size=1,
                                            stride=1, padding=0, bias=0)),
                            ('bn',   nn.BatchNorm2d(128, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv28 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(128, 256, kernel_size=3,
                                            stride=1, padding=1, bias=0)),
                            ('bn',   nn.BatchNorm2d(256, momentum=0.1,
                                            eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                     ]))
        self.conv29 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(256, self.ylch, kernel_size=1,
                                            stride=1, padding=0, bias=1)),
                     ]))
        self.yolo3 = YOLOLayer(self.anchors[0], self.img_size, self.num_classes)

    def set_bn_params(self, layer, params, ptr):
        # bias
        num_b = layer.bias.numel()
        bn_b  = torch.from_numpy(params[ptr : ptr + num_b]).view_as(layer.bias)
        layer.bias.data.copy_(bn_b)
        ptr  += num_b
    
        # weight
        num_w = layer.weight.numel()
        bn_w  = torch.from_numpy(params[ptr : ptr + num_w]).view_as(layer.weight)
        layer.weight.data.copy_(bn_w)
        ptr  += num_w
    
        # running mean
        num_rm = layer.running_mean.numel()
        bn_rm  = torch.from_numpy(
                        params[ptr : ptr + num_rm]).view_as(layer.running_mean)
        layer.running_mean.copy_(bn_rm)
        ptr   += num_rm
    
        # running var
        num_rv = layer.running_var.numel()
        bn_rv  = torch.from_numpy(
                        params[ptr : ptr + num_rv]).view_as(layer.running_var)
        layer.running_var.copy_(bn_rv)
        ptr   += num_rv

        return ptr

    def set_conv_weights(self, layer, params, ptr):
        num_w  = layer.weight.numel()
        conv_w = torch.from_numpy(
                        params[ptr : ptr + num_w]).view_as(layer.weight)
        layer.weight.data.copy_(conv_w)
        ptr   += num_w

        return ptr
    
    def set_conv_biases(self, layer, params, ptr):
        num_b  = layer.bias.numel()
        conv_b = torch.from_numpy(params[ptr : ptr + num_b]).view_as(layer.bias)
        layer.bias.data.copy_(conv_b)
        ptr   += num_b

        return ptr
    
    def load_darknet_weights(self, weights_path):
        # バイナリファイルを読み込み, 配列にデータを格納
        with open(weights_path, "rb") as f:
            weights = np.fromfile(f, dtype=np.float32)
    
        ptr = 5        # 0~4 は, ヘッダのようなものが入っている
    
        ptr = self.set_bn_params(self.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res1.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res1.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res1.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res1.conv2.conv, weights, ptr)

        ptr = self.set_bn_params(self.conv3.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv3.conv, weights, ptr)
        ptr = self.set_bn_params(self.res2.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res2.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res2.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res2.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res3.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res3.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res3.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res3.conv2.conv, weights, ptr)

        ptr = self.set_bn_params(self.conv4.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv4.conv, weights, ptr)
        ptr = self.set_bn_params(self.res4.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res4.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res4.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res4.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res5.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res5.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res5.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res5.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res6.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res6.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res6.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res6.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res7.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res7.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res7.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res7.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res8.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res8.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res8.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res8.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res9.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res9.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res9.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res9.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res10.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res10.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res10.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res10.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res11.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res11.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res11.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res11.conv2.conv, weights, ptr)

        ptr = self.set_bn_params(self.conv5.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv5.conv, weights, ptr)
        ptr = self.set_bn_params(self.res12.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res12.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res12.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res12.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res13.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res13.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res13.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res13.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res14.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res14.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res14.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res14.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res15.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res15.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res15.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res15.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res16.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res16.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res16.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res16.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res17.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res17.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res17.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res17.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res18.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res18.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res18.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res18.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res19.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res19.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res19.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res19.conv2.conv, weights, ptr)

        ptr = self.set_bn_params(self.conv6.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv6.conv, weights, ptr)
        ptr = self.set_bn_params(self.res20.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res20.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res20.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res20.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res21.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res21.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res21.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res21.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res22.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res22.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res22.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res22.conv2.conv, weights, ptr)
        ptr = self.set_bn_params(self.res23.conv1.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res23.conv1.conv, weights, ptr)
        ptr = self.set_bn_params(self.res23.conv2.bn, weights, ptr)
        ptr = self.set_conv_weights(self.res23.conv2.conv, weights, ptr)
    
        ptr = self.set_bn_params(self.conv7.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv7.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv8.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv8.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv9.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv9.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv10.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv10.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv11.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv11.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv12.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv12.conv, weights, ptr)
        ptr = self.set_conv_biases(self.conv13.conv, weights, ptr)
        ptr = self.set_conv_weights(self.conv13.conv, weights, ptr)

        ptr = self.set_bn_params(self.conv14.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv14.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv15.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv15.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv16.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv16.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv17.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv17.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv18.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv18.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv19.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv19.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv20.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv20.conv, weights, ptr)
        ptr = self.set_conv_biases(self.conv21.conv, weights, ptr)
        ptr = self.set_conv_weights(self.conv21.conv, weights, ptr)

        ptr = self.set_bn_params(self.conv22.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv22.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv23.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv23.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv24.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv24.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv25.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv25.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv26.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv26.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv27.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv27.conv, weights, ptr)
        ptr = self.set_bn_params(self.conv28.bn, weights, ptr)
        ptr = self.set_conv_weights(self.conv28.conv, weights, ptr)
        ptr = self.set_conv_biases(self.conv29.conv, weights, ptr)
        ptr = self.set_conv_weights(self.conv29.conv, weights, ptr)

    def forward(self, x):
        yolo_outputs = []

        # backborn
        x = self.conv1(x)

        x = self.conv2(x)       # down sample (416 -> 208)
        x = self.res1(x)

        x = self.conv3(x)       # down sample (208 -> 104)
        x = self.res2(x)
        x = self.res3(x)

        x = self.conv4(x)       # down sample (104 -> 52)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        res11_out = self.res11(x)

        x = self.conv5(res11_out)       # down sample (52 -> 26)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)
        x = self.res17(x)
        x = self.res18(x)
        res19_out = self.res19(x)

        x = self.conv6(res19_out)       # down sample (26 -> 13)
        x = self.res20(x)
        x = self.res21(x)
        x = self.res22(x)
        x = self.res23(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        conv11_out = self.conv11(x)
        x1 = self.conv12(conv11_out)
        x1 = self.conv13(x1)

        # YOLOレイヤ1
        x1 = self.yolo1(x1)
        yolo_outputs.append(x1)
        
        # route (-4)
        x = conv11_out
        x = self.conv14(x)
        x = self.upsample1(x)

        # route (-1, 61) : concat
        x = torch.cat([x, res19_out], dim=1)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        conv19_out = self.conv19(x)
        x2 = self.conv20(conv19_out)
        x2 = self.conv21(x2)

        # YOLOレイヤ2
        x2 = self.yolo2(x2)
        yolo_outputs.append(x2)
        
        # route (-4)
        x = conv19_out
        x = self.conv22(x)
        x = self.upsample2(x)

        # route (-1, 36) : concat
        x = torch.cat([x, res11_out], dim=1)
        x = self.conv23(x)
        x = self.conv24(x)
        x = self.conv25(x)
        x = self.conv26(x)
        x = self.conv27(x)
        x = self.conv28(x)
        x = self.conv29(x)
        # YOLOレイヤ2
        x3 = self.yolo3(x)
        yolo_outputs.append(x3)

        if self.training:
            return yolo_outputs
        return torch.cat(yolo_outputs, 1)

class YOLO_sep(nn.Module):
    def __init__(self, num_classes):
        super(YOLO_sep, self).__init__()
        #self.anchors     = [[[10,14], [23,27], [37,58]],
        #                    [[81,82], [135,169], [344,319]]]
        self.anchors     = [[[23,27], [37,58], [81,82]],
                            [[81,82], [135,169], [344,319]]]
        self.img_size    = 416
        self.num_classes = num_classes
        self.ylch        = (5 + self.num_classes) * 3       # yolo layer channels

        # modules
        # 1層目を separable にしても意外と精度落ちないので採用
        self.conv1  = nn.Sequential(OrderedDict([
                           ('conv_dw', nn.Conv2d(   3,    3, kernel_size=3,
                               groups=3,   stride=1, padding=1, bias=0)),  # dw
                           ('bn_dw', nn.BatchNorm2d(  3, momentum=0.1, eps=1e-5)),
                           ('relu_dw', nn.LeakyReLU(0.1)),
                           ('conv_pw', nn.Conv2d(   3,   16, kernel_size=1,
                               stride=1, padding=0, bias=0)),              # pw
                           ('bn_pw', nn.BatchNorm2d(  16, momentum=0.1, eps=1e-5)),
                           ('relu_pw', nn.LeakyReLU(0.1))
                           ]))
        # 1層目のみ, 普通の3x3-conv
        #self.conv1  = nn.Sequential(OrderedDict([
        #                    ('conv', nn.Conv2d(   3,   16, kernel_size=3,
        #                        stride=1, padding=1, bias=0)),
        #                    ('bn', nn.BatchNorm2d(  16, momentum=0.1, eps=1e-5)),
        #                    ('relu', nn.LeakyReLU(0.1))
        #                    ]))
        self.conv2  = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d(  16,   16, kernel_size=3,
                                groups=16,  stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d(  16, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d(  16,   32, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d(  32, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv3  = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d(  32,   32, kernel_size=3,
                                groups=32,  stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d(  32, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d(  32,   64, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d(  64, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv4  = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d(  64,   64, kernel_size=3,
                                groups=64,  stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d( 64, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d(  64,  128, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d( 128, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv5  = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d( 128,  128, kernel_size=3,
                                groups=128, stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d( 128, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d( 128,  256, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv6  = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d( 256,  256, kernel_size=3,
                                groups=256, stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d( 256,  512, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d( 512, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv7  = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d( 512,  512, kernel_size=3,
                                groups=512, stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d(512, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d( 512, 1024, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d(1024, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv8  = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(1024,  256, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn', nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                            ]))
        self.conv9  = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d( 256,  256, kernel_size=3,
                                groups=256, stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d( 256,  512, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d( 512, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv10 = nn.Conv2d(512, self.ylch, kernel_size=1,
                                stride=1, padding=0, bias=1)
        self.conv11 = nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d( 256,  128, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn', nn.BatchNorm2d( 128, momentum=0.1, eps=1e-5)),
                            ('relu', nn.LeakyReLU(0.1))
                            ]))
        self.conv12 = nn.Sequential(OrderedDict([
                            ('conv_dw', nn.Conv2d( 384,  384, kernel_size=3,
                                groups=384, stride=1, padding=1, bias=0)),
                            ('bn_dw', nn.BatchNorm2d( 384, momentum=0.1, eps=1e-5)),
                            ('relu_dw', nn.LeakyReLU(0.1)),
                            ('conv_pw', nn.Conv2d( 384,  256, kernel_size=1,
                                stride=1, padding=0, bias=0)),
                            ('bn_pw', nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)),
                            ('relu_pw', nn.LeakyReLU(0.1))
                            ]))
        self.conv13   = nn.Conv2d(256, self.ylch, kernel_size=1,
                                stride=1, padding=0, bias=1)
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool5    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool6    = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.zeropad  = nn.ZeroPad2d((0, 1, 0, 1))
                                # サイズを保つためのゼロパディング (pool6の直前)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.yolo1    = YOLOLayer(self.anchors[1],
                            self.img_size, self.num_classes)
        self.yolo2    = YOLOLayer(self.anchors[0],
                            self.img_size, self.num_classes)

        self.yolo_layers = [self.yolo1, self.yolo2]

    def forward(self, x, distri_array=None, debug=False):
        yolo_outputs = []

        step1_t = time.time()
        # 特徴抽出部
        #if debug: print("input:", x[0][0][200])
        x = self.conv1(x)
        if debug: print("conv1 output :", x)
        #if debug: print("conv1 max :", torch.max(x))
        #if debug: print("conv1 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.pool1(x)
        x = self.conv2(x)
        if debug: print("conv2 max :", torch.max(x))
        if debug: print("conv2 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.pool2(x)
        x = self.conv3(x)
        if debug: print("conv3 max :", torch.max(x))
        if debug: print("conv3 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.pool3(x)
        x = self.conv4(x)
        if debug: print("conv4 max :", torch.max(x))
        if debug: print("conv4 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.pool4(x)
        x = self.conv5(x)
        if debug: print("conv5 max :", torch.max(x))
        if debug: print("conv5 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        l8_output = x       # あとのconcat用に出力を保管
        x = self.pool5(x)
        x = self.conv6(x)
        if debug: print("conv6 max :", torch.max(x))
        if debug: print("conv6 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x = self.zeropad(x)
        x = self.pool6(x)

        step2_t = time.time()
        # スケール大 検出部
        x = self.conv7(x)
        if debug: print("conv7 max :", torch.max(x))
        if debug: print("conv7 min :", torch.min(x))
        if distri_array is not None: ditsrib.count(distri_array, x)
        x1 = self.conv8(x)
        if debug: print("conv8 max :", torch.max(x1))
        if debug: print("conv8 min :", torch.min(x1))
        if distri_array is not None: ditsrib.count(distri_array, x1)
        x2 = x1
        x1 = self.conv9(x1)
        if debug: print("conv9 max :", torch.max(x1))
        if debug: print("conv9 min :", torch.min(x1))
        if distri_array is not None: ditsrib.count(distri_array, x1)
        x1 = self.conv10(x1)
        if debug: print("conv10 max :", torch.max(x1))
        if debug: print("conv10 min :", torch.min(x1))
        if distri_array is not None: ditsrib.count(distri_array, x1)
        x1 = self.yolo1(x1)
        yolo_outputs.append(x1)

        step3_t = time.time()
        # スケール中 検出部
        x2 = self.conv11(x2)
        if distri_array is not None: ditsrib.count(distri_array, x2)
        x2 = self.upsample(x2)
        # チャネル数方向に特徴マップを結合
        x2 = torch.cat([x2, l8_output], dim=1)
        x2 = self.conv12(x2)
        if distri_array is not None: ditsrib.count(distri_array, x2)
        x2 = self.conv13(x2)
        if distri_array is not None: ditsrib.count(distri_array, x2)
        x2 = self.yolo2(x2)
        yolo_outputs.append(x2)

        last_t = time.time()

        #if not self.training:
        #    print("step1 : %.4f, step2 : %.4f, step3 : %.4f" % 
        #            (step2_t - step1_t, step3_t - step2_t, last_t - step3_t))
        if self.training:
            return yolo_outputs
        return torch.cat(yolo_outputs, 1)

class YOLOLayer(nn.Module):
    def __init__(self, anchors, img_size, num_classes=80):
        super(YOLOLayer, self).__init__()
        self.anchors     = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size    = img_size             # 416
        self.stride      = 0

    def forward(self, x):
        # ストライドは, 画像サイズをグリッドの分割数で割った値.
        # つまり, 1グリッドあたり何ピクセルか　を表す
        self.stride = self.img_size // x.size(2)
        batch_size, _, height, width = x.shape

        # (batch数, 255, H, W) -> (batch数, 3, 85, H, W)
        #                      -> (batch数, 3, H, W, 85)
        # 3 は アンカー数, 85 は x,y,w,h + クラス数
        x = x.reshape(batch_size, self.num_anchors,
                            self.num_classes+5, width, height)
        x = x.permute(0, 1, 3, 4, 2)


        # 推論のときは, ロジステック回帰をして, 座標値やconfidenceに変換して返す
        # ... は, 次元省略
        if not self.training:
            # ボックスの座標のオフセットテーブルを作成
            grid_y, grid_x = torch.meshgrid(
                        torch.arange(height, device=x.device),
                        torch.arange(width, device=x.device)
            )
            # xy
            x[..., 0] = (x[..., 0].sigmoid() + grid_x) * self.stride
            x[..., 1] = (x[..., 1].sigmoid() + grid_y) * self.stride

            anchors = torch.tensor(self.anchors).float()

            # wh
            # anchors は, [[10,14], [23,27], [37,58]] のようなリスト
            # anchors_w, anchors_h を, かける対象のxのshape(1, 3, 13, 13, 1)
            # に合わせる (xは [1, 3, 13, 13, 85] だが, インデックス2 を指定する
            # ので, [1, 3, 13, 13, 1] になる
            anchors_w = torch.tensor([anc[0] for anc in anchors]).to(x.device)
            anchors_w = anchors_w.view(1, -1, 1, 1)
            anchors_h = torch.tensor([anc[1] for anc in anchors]).to(x.device)
            anchors_h = anchors_h.view(1, -1, 1, 1)
            x[..., 2] = torch.exp(x[..., 2]) * anchors_w
            x[..., 3] = torch.exp(x[..., 3]) * anchors_h

            # conf, class確率
            x[..., 4:] = x[..., 4:].sigmoid()

            x = x.reshape(batch_size, -1, self.num_classes+5)

        # 学習のときは, xをそのまま返す. 推論のときは, 変換した値を返す
        return x

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch

        self.conv1 = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(self.ch, self.ch//2, kernel_size=1,
                                        padding=0, stride=1)),
                        ('bn',   nn.BatchNorm2d(self.ch//2, momentum=0.1,
                                        eps=1e-5)),
                        ('relu', nn.LeakyReLU(0.1))
                     ]))
                
        self.conv2 = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(self.ch//2, self.ch, kernel_size=3,
                                        padding=1, stride=1)),
                        ('bn',   nn.BatchNorm2d(self.ch, momentum=0.1,
                                        eps=1e-5)),
                        ('relu', nn.LeakyReLU(0.1))
                     ]))

    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f)

        return f + x

def load_model(weights_path, device, num_classes=80, tiny=True, trans=False,
               restart=False, finetune=False, use_sep=False, quant=False,
               dropout=False, jit=False):
    model = None

    # YOLO-tiny model
    if tiny:
        if trans:
          param_to_update = []
          update_param_names = ['conv10.conv.weight', 'conv10.conv.bias',
                                'conv13.conv.weight', 'conv13.conv.bias']

          if not restart:
              model = YOLO_tiny(80, dropout).to(device)
          else:
              model = YOLO_tiny(num_classes, dropout).to(device)

          if weights_path.endswith('weights'):
              model.load_darknet_weights(weights_path);
          else:
              #model.load_weights(weights_path, device)
              model.load_state_dict(torch.load(
                                        weights_path, map_location=device))

          # 最終層を置き換え
          if not restart:
              ylch = (5 + num_classes) * 3
              model.conv10.conv = nn.Conv2d(512, ylch, kernel_size=1,
                                    stride=1, padding=0, bias=1)
              model.conv13.conv = nn.Conv2d(256, ylch, kernel_size=1,
                                    stride=1, padding=0, bias=1)
              model.yolo1       = YOLOLayer(model.anchors[1],
                                    model.img_size, num_classes)
              model.yolo2       = YOLOLayer(model.anchors[0],
                                    model.img_size, num_classes)
              model.yolo_layers = [model.yolo1, model.yolo2]

          # 置き換えた層以外のパラメータをフリーズ
          for (key, param) in model.named_parameters():
            if key in update_param_names:
              param.requires_grad = True
              param_to_update.append(param)
            else:
              param.requires_grad = False

          model.to(device)
          return model, param_to_update

        elif finetune:
          model = (YOLO_sep(80).to(device) 
                        if use_sep 
                        else YOLO_tiny(80, dropout).to(device))
          if restart:
            weights = torch.load(weights_path, map_location=device)
            model.load_state_dict(weights)
          else:
            if weights_path.endswith('weights'):
                model.load_darknet_weights(weights_path);
            else:
                #model.load_weights(weights_path, device)
                weights = torch.load(weights_path, map_location=device)
                model.load_state_dict(weights)
            # 最終層を置き換え
            ylch = (5 + num_classes) * 3
            model.conv10.conv = nn.Conv2d(512, ylch, kernel_size=1,
                                    stride=1, padding=0, bias=1)
            model.conv13.conv = nn.Conv2d(256, ylch, kernel_size=1,
                                    stride=1, padding=0, bias=1)
            model.yolo1       = YOLOLayer(model.anchors[1],
                                    model.img_size, num_classes)
            model.yolo2       = YOLOLayer(model.anchors[0],
                                    model.img_size, num_classes)
            model.yolo_layers = [model.yolo1, model.yolo2]

            model.to(device)
          return model

        else:           # 推論 or 一から学習
          model = (YOLO_sep(num_classes).to(device)
                    if use_sep
                    else YOLO_tiny(num_classes, quant, dropout).to(device))

          if weights_path:
            if quant and jit:               # 量子化モデルを使った推論
                model = torch.jit.load(weights_path)
            elif weights_path.endswith('weights'):
                model.load_darknet_weights(weights_path)
            else:       # pt file
                model.load_state_dict(torch.load(
                            weights_path, map_location=device))

    # YOLO model
    else:               
        model = YOLO(80).to(device)
        model.load_darknet_weights(weights_path)


    #print(model)
    return model
