import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# YOLOv3-tiny
class YOLO(nn.Module):
    def __init__(self, num_classes):
        super(YOLO, self).__init__()
        self.anchors     = [[[10,14], [23,27], [37,58]], [[81,82], [135,169], [344,319]]]
        self.img_size    = 416
        self.num_classes = num_classes
        self.ylch        = (5 + self.num_classes) * 3       # yolo layer channels

        # modules
        self.conv1    = nn.Conv2d(   3,   16, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv2    = nn.Conv2d(  16,   32, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv3    = nn.Conv2d(  32,   64, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv4    = nn.Conv2d(  64,  128, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv5    = nn.Conv2d( 128,  256, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv6    = nn.Conv2d( 256,  512, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv7    = nn.Conv2d( 512, 1024, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv8    = nn.Conv2d(1024,  256, kernel_size=1, stride=1, padding=0, bias=0)
        self.conv9    = nn.Conv2d( 256,  512, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv10   = nn.Conv2d( 512, self.ylch, kernel_size=1, stride=1, padding=0, bias=1)
        self.conv11   = nn.Conv2d( 256,  128, kernel_size=1, stride=1, padding=0, bias=0)
        self.conv12   = nn.Conv2d( 384,  256, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv13   = nn.Conv2d( 256, self.ylch, kernel_size=1, stride=1, padding=0, bias=1)
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool5    = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool6    = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.zeropad  = nn.ZeroPad2d((0, 1, 0, 1))      # $B%5%$%:$rJ]$D$?$a$N%<%m%Q%G%#%s%0(B (pool6$B$ND>A0(B)
        self.bn1      = nn.BatchNorm2d(  16, momentum=0.1, eps=1e-5)
        self.bn2      = nn.BatchNorm2d(  32, momentum=0.1, eps=1e-5)
        self.bn3      = nn.BatchNorm2d(  64, momentum=0.1, eps=1e-5)
        self.bn4      = nn.BatchNorm2d( 128, momentum=0.1, eps=1e-5)
        self.bn5      = nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)
        self.bn6      = nn.BatchNorm2d( 512, momentum=0.1, eps=1e-5)
        self.bn7      = nn.BatchNorm2d(1024, momentum=0.1, eps=1e-5)
        self.bn8      = nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)
        self.bn9      = nn.BatchNorm2d( 512, momentum=0.1, eps=1e-5)
        self.bn10     = nn.BatchNorm2d( 128, momentum=0.1, eps=1e-5)
        self.bn11     = nn.BatchNorm2d( 256, momentum=0.1, eps=1e-5)
        self.relu     = nn.LeakyReLU(0.1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.yolo1    = YOLOLayer(self.anchors[1], self.img_size, self.num_classes)     # $B$"$H$G%/%i%9:n$k(B
        self.yolo2    = YOLOLayer(self.anchors[0], self.img_size, self.num_classes)

        self.yolo_layers = [self.yolo1, self.yolo2]

        # hyperparams

    def load_weights(self, weights_path, device):
        ckpt = torch.load(weights_path, map_location=device)
        param = ckpt['model']

        self.conv1.weight     = nn.Parameter(param['module_list.0.Conv2d.weight'])
        self.bn1.weight       = nn.Parameter(param['module_list.0.BatchNorm2d.weight'])
        self.bn1.bias         = nn.Parameter(param['module_list.0.BatchNorm2d.bias'])
        self.bn1.running_mean = param['module_list.0.BatchNorm2d.running_mean']
        self.bn1.running_var  = param['module_list.0.BatchNorm2d.running_var']
        self.bn1.num_batches_tracked = param['module_list.0.BatchNorm2d.num_batches_tracked']
        self.conv2.weight     = nn.Parameter(param['module_list.2.Conv2d.weight'])
        self.bn2.weight       = nn.Parameter(param['module_list.2.BatchNorm2d.weight'])
        self.bn2.bias         = nn.Parameter(param['module_list.2.BatchNorm2d.bias'])
        self.bn2.running_mean = param['module_list.2.BatchNorm2d.running_mean']
        self.bn2.running_var  = param['module_list.2.BatchNorm2d.running_var']
        self.bn2.num_batches_tracked = param['module_list.2.BatchNorm2d.num_batches_tracked']
        self.conv3.weight     = nn.Parameter(param['module_list.4.Conv2d.weight'])
        self.bn3.weight       = nn.Parameter(param['module_list.4.BatchNorm2d.weight'])
        self.bn3.bias         = nn.Parameter(param['module_list.4.BatchNorm2d.bias'])
        self.bn3.running_mean = param['module_list.4.BatchNorm2d.running_mean']
        self.bn3.running_var  = param['module_list.4.BatchNorm2d.running_var']
        self.bn3.num_batches_tracked = param['module_list.4.BatchNorm2d.num_batches_tracked']
        self.conv4.weight     = nn.Parameter(param['module_list.6.Conv2d.weight'])
        self.bn4.weight       = nn.Parameter(param['module_list.6.BatchNorm2d.weight'])
        self.bn4.bias         = nn.Parameter(param['module_list.6.BatchNorm2d.bias'])
        self.bn4.running_mean = param['module_list.6.BatchNorm2d.running_mean']
        self.bn4.running_var  = param['module_list.6.BatchNorm2d.running_var']
        self.bn4.num_batches_tracked = param['module_list.6.BatchNorm2d.num_batches_tracked']
        self.conv5.weight     = nn.Parameter(param['module_list.8.Conv2d.weight'])
        self.bn5.weight       = nn.Parameter(param['module_list.8.BatchNorm2d.weight'])
        self.bn5.bias         = nn.Parameter(param['module_list.8.BatchNorm2d.bias'])
        self.bn5.running_mean = param['module_list.8.BatchNorm2d.running_mean']
        self.bn5.running_var  = param['module_list.8.BatchNorm2d.running_var']
        self.bn5.num_batches_tracked = param['module_list.8.BatchNorm2d.num_batches_tracked']
        self.conv6.weight     = nn.Parameter(param['module_list.10.Conv2d.weight'])
        self.bn6.weight       = nn.Parameter(param['module_list.10.BatchNorm2d.weight'])
        self.bn6.bias         = nn.Parameter(param['module_list.10.BatchNorm2d.bias'])
        self.bn6.running_mean = param['module_list.10.BatchNorm2d.running_mean']
        self.bn6.running_var  = param['module_list.10.BatchNorm2d.running_var']
        self.bn6.num_batches_tracked = param['module_list.10.BatchNorm2d.num_batches_tracked']

        self.conv7.weight     = nn.Parameter(param['module_list.12.Conv2d.weight'])
        self.bn7.weight       = nn.Parameter(param['module_list.12.BatchNorm2d.weight'])
        self.bn7.bias         = nn.Parameter(param['module_list.12.BatchNorm2d.bias'])
        self.bn7.running_mean = param['module_list.12.BatchNorm2d.running_mean']
        self.bn7.running_var  = param['module_list.12.BatchNorm2d.running_var']
        self.bn7.num_batches_tracked = param['module_list.12.BatchNorm2d.num_batches_tracked']
        self.conv8.weight     = nn.Parameter(param['module_list.13.Conv2d.weight'])
        self.bn8.weight       = nn.Parameter(param['module_list.13.BatchNorm2d.weight'])
        self.bn8.bias         = nn.Parameter(param['module_list.13.BatchNorm2d.bias'])
        self.bn8.running_mean = param['module_list.13.BatchNorm2d.running_mean']
        self.bn8.running_var  = param['module_list.13.BatchNorm2d.running_var']
        self.bn8.num_batches_tracked = param['module_list.13.BatchNorm2d.num_batches_tracked']
        self.conv9.weight     = nn.Parameter(param['module_list.14.Conv2d.weight'])
        self.bn9.weight       = nn.Parameter(param['module_list.14.BatchNorm2d.weight'])
        self.bn9.bias         = nn.Parameter(param['module_list.14.BatchNorm2d.bias'])
        self.bn9.running_mean = param['module_list.14.BatchNorm2d.running_mean']
        self.bn9.running_var  = param['module_list.14.BatchNorm2d.running_var']
        self.bn9.num_batches_tracked = param['module_list.14.BatchNorm2d.num_batches_tracked']

        self.conv10.weight    = nn.Parameter(param['module_list.15.Conv2d.weight'])
        self.conv10.bias      = nn.Parameter(param['module_list.15.Conv2d.bias'])

        self.conv11.weight     = nn.Parameter(param['module_list.18.Conv2d.weight'])
        self.bn10.weight       = nn.Parameter(param['module_list.18.BatchNorm2d.weight'])
        self.bn10.bias         = nn.Parameter(param['module_list.18.BatchNorm2d.bias'])
        self.bn10.running_mean = param['module_list.18.BatchNorm2d.running_mean']
        self.bn10.running_var  = param['module_list.18.BatchNorm2d.running_var']
        self.bn10.num_batches_tracked = param['module_list.18.BatchNorm2d.num_batches_tracked']
        self.conv12.weight     = nn.Parameter(param['module_list.21.Conv2d.weight'])
        self.bn11.weight       = nn.Parameter(param['module_list.21.BatchNorm2d.weight'])
        self.bn11.bias         = nn.Parameter(param['module_list.21.BatchNorm2d.bias'])
        self.bn11.running_mean = param['module_list.21.BatchNorm2d.running_mean']
        self.bn11.running_var  = param['module_list.21.BatchNorm2d.running_var']
        self.bn11.num_batches_tracked = param['module_list.21.BatchNorm2d.num_batches_tracked']

        self.conv13.weight    = nn.Parameter(param['module_list.22.Conv2d.weight'])
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
        bn_rm  = torch.from_numpy(params[ptr : ptr + num_rm]).view_as(layer.running_mean)
        layer.running_mean.copy_(bn_rm)
        ptr   += num_rm
    
        # running var
        num_rv = layer.running_var.numel()
        bn_rv  = torch.from_numpy(params[ptr : ptr + num_rv]).view_as(layer.running_var)
        layer.running_var.copy_(bn_rv)
        ptr   += num_rv

    def set_conv_weights(self, layer, params, ptr):
        num_w  = layer.weight.numel()
        conv_w = torch.from_numpy(params[ptr : ptr + num_w]).view_as(layer.weight)
        layer.weight.data.copy_(conv_w)
        ptr   += num_w
    
    def set_conv_biases(self, layer, params, ptr):
        num_b  = layer.bias.numel()
        conv_b = torch.from_numpy(params[ptr : ptr + num_b]).view_as(layer.bias)
        layer.bias.data.copy_(conv_b)
        ptr   += num_b
    
    def load_darknet_weights(self, weights_path):
        # $B%P%$%J%j%U%!%$%k$rFI$_9~$_(B, $BG[Ns$K%G!<%?$r3JG<(B
        with open(weights_path, "rb") as f:
            # skip header
            f.read(20)

            weights = np.fromfile(f, dtype=np.float32)
    
        ptr = 0
    
        self.set_bn_params(self.bn1, weights, ptr)
        self.set_conv_weights(self.conv1, weights, ptr)
        self.set_bn_params(self.bn2, weights, ptr)
        self.set_conv_weights(self.conv2, weights, ptr)
        self.set_bn_params(self.bn3, weights, ptr)
        self.set_conv_weights(self.conv3, weights, ptr)
        self.set_bn_params(self.bn4, weights, ptr)
        self.set_conv_weights(self.conv4, weights, ptr)
        self.set_bn_params(self.bn5, weights, ptr)
        self.set_conv_weights(self.conv5, weights, ptr)
        self.set_bn_params(self.bn6, weights, ptr)
        self.set_conv_weights(self.conv6, weights, ptr)
    
        self.set_bn_params(self.bn7, weights, ptr)
        self.set_conv_weights(self.conv7, weights, ptr)
        self.set_bn_params(self.bn8, weights, ptr)
        self.set_conv_weights(self.conv8, weights, ptr)
        self.set_bn_params(self.bn9, weights, ptr)
        self.set_conv_weights(self.conv9, weights, ptr)
    
        self.set_conv_biases(self.conv10, weights, ptr)
        self.set_conv_weights(self.conv10, weights, ptr)
    
        self.set_bn_params(self.bn10, weights, ptr)
        self.set_conv_weights(self.conv11, weights, ptr)
        self.set_bn_params(self.bn11, weights, ptr)
        self.set_conv_weights(self.conv12, weights, ptr)
    
        self.set_conv_biases(self.conv13, weights, ptr)
        self.set_conv_weights(self.conv13, weights, ptr)

    def forward(self, x):
        yolo_outputs = []

        # $BFCD'Cj=PIt(B
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        l8_output = x       # $B$"$H$N(Bconcat$BMQ$K=PNO$rJ]4I(B
        x = self.pool5(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.zeropad(x)
        x = self.pool6(x)

        # $B%9%1!<%kBg(B $B8!=PIt(B
        x = self.relu(self.bn7(self.conv7(x)))
        x1 = self.relu(self.bn8(self.conv8(x)))
        x2 = x1
        x1 = self.relu(self.bn9(self.conv9(x1)))
        x1 = self.conv10(x1)
        x1 = self.yolo1(x1)
        yolo_outputs.append(x1)

        # $B%9%1!<%kCf(B $B8!=PIt(B
        x2 = self.relu(self.bn10(self.conv11(x2)))
        x2 = self.upsample(x2)
        x2 = torch.cat([x2, l8_output], dim=1)        # $B%A%c%M%k?tJ}8~$KFCD'%^%C%W$r7k9g(B
        x2 = self.relu(self.bn11(self.conv12(x2)))
        x2 = self.conv13(x2)
        x2 = self.yolo2(x2)
        yolo_outputs.append(x2)

        # $B$?$V$s(Bself.training$B$O(B, model.train() $B$K$7$?;~E@$G(BTrue$B$K$J$k(B
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)

class YOLOLayer(nn.Module):
    def __init__(self, anchors, img_size, num_classes=80):
        super(YOLOLayer, self).__init__()
        self.anchors     = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size    = img_size             # 416
        self.stride      = None

    def forward(self, x):
        # $B%9%H%i%$%I$O(B, $B2hA|%5%$%:$r%0%j%C%I$NJ,3d?t$G3d$C$?CM(B.
        # $B$D$^$j(B, 1$B%0%j%C%I$"$?$j2?%T%/%;%k$+!!$rI=$9(B
        self.stride = self.img_size // x.size(2)
        batch_size, _, height, width = x.shape

        # (batch$B?t(B, 255, H, W) -> (batch$B?t(B, 3, 85, H, W) -> (batch$B?t(B, 3, H, W, 85)
        # 3 $B$O(B $B%"%s%+!<?t(B, 85 $B$O(B x,y,w,h + $B%/%i%9?t(B
        x = x.reshape(batch_size, self.num_anchors, self.num_classes+5, width, height)
        x = x.permute(0, 1, 3, 4, 2)


        # $B?dO@$N$H$-$O(B, $B%m%8%9%F%C%/2s5"$r$7$F(B, $B:BI8CM$d(Bconfidence$B$KJQ49$7$FJV$9(B
        # ... $B$O(B, $B<!85>JN,(B
        if not self.training:
            # $B%\%C%/%9$N:BI8$N%*%U%;%C%H%F!<%V%k$r:n@.(B
            grid_y, grid_x = torch.meshgrid(
                        torch.arange(height, device=x.device),
                        torch.arange(width, device=x.device)
            )
            # xy
            x[..., 0] = (x[..., 0].sigmoid() + grid_x) * self.stride
            x[..., 1] = (x[..., 1].sigmoid() + grid_y) * self.stride

            anchors = torch.tensor(self.anchors).float()

            # wh
            # anchors $B$O(B, [[10,14], [23,27], [37,58]] $B$N$h$&$J%j%9%H(B
            # anchors_w, anchors_h $B$r(B, $B$+$1$kBP>]$N(Bx$B$N(Bshape(1, 3, 13, 13, 1)
            # $B$K9g$o$;$k(B (x$B$O(B [1, 3, 13, 13, 85] $B$@$,(B, $B%$%s%G%C%/%9(B2 $B$r;XDj$9$k(B
            # $B$N$G(B, [1, 3, 13, 13, 1] $B$K$J$k(B
            anchors_w = torch.tensor([anc[0] for anc in anchors]).to(x.device)
            anchors_w = anchors_w.view(1, -1, 1, 1)
            anchors_h = torch.tensor([anc[1] for anc in anchors]).to(x.device)
            anchors_h = anchors_h.view(1, -1, 1, 1)
            x[..., 2] = torch.exp(x[..., 2]) * anchors_w
            x[..., 3] = torch.exp(x[..., 3]) * anchors_h

            # conf, class$B3NN((B
            x[..., 4:] = x[..., 4:].sigmoid()

            x = x.reshape(batch_size, -1, self.num_classes+5)

        # $B3X=,$N$H$-$O(B, x$B$r$=$N$^$^JV$9(B. $B?dO@$N$H$-$O(B, $BJQ49$7$?CM$rJV$9(B
        return x

def load_model(weights_path, device, num_classes=80, trans=False):
    model = None

    param_to_update = []
    if trans:
      update_param_names = ['conv10.weight', 'conv10.bias',
                            'conv13.weight', 'conv13.bias']

      model = YOLO(80).to(device)
      model.load_weights(weights_path, device)

      # $B:G=*AX$rCV$-49$((B
      ylch = (5 + num_classes) * 3
      model.conv10 = nn.Conv2d(512, ylch, kernel_size=1, stride=1, padding=0, bias=1)
      model.conv13 = nn.Conv2d(256, ylch, kernel_size=1, stride=1, padding=0, bias=1)
      model.yolo1 = YOLOLayer(model.anchors[1], model.img_size, num_classes)
      model.yolo2 = YOLOLayer(model.anchors[0], model.img_size, num_classes)
      model.yolo_layers = [model.yolo1, model.yolo2]

      # $BCV$-49$($?AX0J30$N%Q%i%a!<%?$r%U%j!<%:(B
      for (key, param) in model.named_parameters():
        if key in update_param_names:
          param.requires_grad = True
          param_to_update.append(param)
        else:
          param.requires_grad = False

      model.to(device)
      return model, param_to_update

    else:
      model = YOLO(num_classes).to(device)
      if weights_path:
        if weights_path.endswith('weights'):
            model.load_darknet_weights(weights_path)
        else:       # pt file
            model = YOLO(num_classes).to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device))

      return model
