import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# YOLOv3-tiny
class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.anchors     = [[[10,14], [23,27], [37,58]], [[81,82], [135,169], [344,319]]]
        self.img_size    = 416
        self.num_classes = 80

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
        self.conv10   = nn.Conv2d( 512,  255, kernel_size=1, stride=1, padding=0, bias=0)
        self.conv11   = nn.Conv2d( 256,  128, kernel_size=1, stride=1, padding=0, bias=0)
        self.conv12   = nn.Conv2d( 384,  256, kernel_size=3, stride=1, padding=1, bias=0)
        self.conv13   = nn.Conv2d( 256,  255, kernel_size=1, stride=1, padding=0, bias=0)
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
            x[..., 0] = x[..., 0].sigmoid() + grid_x
            x[..., 1] = x[..., 1].sigmoid() + grid_y

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
