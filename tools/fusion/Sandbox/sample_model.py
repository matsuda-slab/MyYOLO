import torch
import torch.nn as nn

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv_dw = nn.Conv2d(3, 3, kernel_size=3, groups=3, stride=1, padding=1, bias=0)
        self.bn_dw = nn.BatchNorm2d(3, momentum=0.1, eps=1e-5)
        self.conv_pw = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=0)
        self.bn_pw = nn.BatchNorm2d(16, momentum=0.1, eps=1e-5)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        #print(x)
        x = self.conv_pw(x)
        x = self.bn_pw(x)
        
        return x

class SampleNet_merge(nn.Module):
    def __init__(self):
        super(SampleNet_merge, self).__init__()
        self.conv_dw = nn.Conv2d(3, 3, kernel_size=3, groups=3, stride=1, padding=1, bias=1)
        self.conv_pw = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=1)

    def forward(self, x):
        x = self.conv_dw(x)
        #print(x)
        x = self.conv_pw(x)

        return x

def main():
    weights_path = 'weights/nomerge.pt'
    weights_merge_path = 'weights/merge.pt'

    weights = torch.load(weights_path, map_location='cpu')
    weights_merge = torch.load(weights_merge_path, map_location='cpu')

    model = SampleNet()
    model_merge = SampleNet_merge()

    # パラメータファイルで重みの初期化
    model.state_dict()['conv_dw.weight'][:] = weights['conv1.conv_dw.weight']
    model.state_dict()['bn_dw.weight'][:] = weights['conv1.bn_dw.weight']
    #print(model.state_dict()['bn_dw.weight'][0])
    model.state_dict()['bn_dw.bias'][:] = weights['conv1.bn_dw.bias']
    model.state_dict()['bn_dw.running_mean'][:] = weights['conv1.bn_dw.running_mean']
    model.state_dict()['bn_dw.running_var'][:] = weights['conv1.bn_dw.running_var']
    #print(model.state_dict()['bn_dw.running_var'][0])
    model.state_dict()['conv_pw.weight'][:] = weights['conv1.conv_pw.weight']
    model.state_dict()['bn_pw.weight'][:] = weights['conv1.bn_pw.weight']
    model.state_dict()['bn_pw.bias'][:] = weights['conv1.bn_pw.bias']
    model.state_dict()['bn_pw.running_mean'][:] = weights['conv1.bn_pw.running_mean']
    model.state_dict()['bn_pw.running_var'][:] = weights['conv1.bn_pw.running_var']

    #print(model_merge.state_dict()['conv_dw.weight'])
    model_merge.state_dict()['conv_dw.weight'][:] = weights['conv1.conv_dw.weight']
    model_merge.state_dict()['conv_dw.bias'][:] = weights['conv1.conv_dw.bias']
    model_merge.state_dict()['conv_pw.weight'][:] = weights['conv1.conv_pw.weight']
    model_merge.state_dict()['conv_pw.bias'][:] = weights['conv1.conv_pw.bias']
    #print(model_merge.state_dict()['conv_dw.weight'])

    input = torch.randn(1, 3, 16, 16)

    model.eval()
    #print(model(input))
    model_merge.eval()
    #print(model_merge(input))

if __name__ == "__main__":
    main()
