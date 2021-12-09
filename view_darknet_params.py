import numpy as np
import torch

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
    bn_rm  = torch.from_numpy(params[ptr : ptr + num_rm]).view_as(layer.running_mean)
    layer.running_mean.copy_(bn_rm)
    ptr   += num_rm

    # running var
    num_rv = layer.running_var.numel()
    bn_rv  = torch.from_numpy(params[ptr : ptr + num_rv]).view_as(layer.running_var)
    layer.running_var.copy_(bn_rv)
    ptr   += num_rv

    return ptr

def set_conv_weights(self, layer, params, ptr):
    num_w  = layer.weight.numel()
    conv_w = torch.from_numpy(params[ptr : ptr + num_w]).view_as(layer.weight)
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
        # skip header
        #f.read(20)

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

if __name__ == "__main__":
    weights_path = 'weights/yolov3-tiny.weights'
    with open(weights_path, "rb") as f:
        weights = np.fromfile(f, dtype=np.float32)
        print(weights[0:100])
