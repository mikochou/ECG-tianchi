import torch
import torch.nn as nn
from torchsummary import summary
from .BasicModule import BasicModule
from IPython import embed


def activation(act_type='prelu'):
    if act_type == 'prelu':
        act = nn.PReLU()
    elif act_type == 'relu':
        act = nn.ReLU(inplace=True)
    else:
        raise NotImplementedError
    return act


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = activation('relu')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)
        #self.dropout = nn.Dropout(.5)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, bias=False, padding=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=11, stride=stride,
                               padding=5, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=7, bias=False, padding=3)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = activation('relu')
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(BasicModule):

    def __init__(self, block, layers, num_classes=55):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(8, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = activation('relu')
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Baseline
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # V1
        #self.avgpool = nn.AdaptiveAvgPool1d(2)
        #self.fc = nn.Linear(2 * 512 * block.expansion, num_classes)
        # v2
        #self.conv2 = nn.Conv1d(2048, 64, kernel_size=1, stride=1, padding=0,
        #                       bias=False)
        #self.bn2 = nn.BatchNorm1d(64)
        #self.avgpool = nn.AdaptiveAvgPool1d(8)
        #self.fc = nn.Linear(64 * 8, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.init()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, data):
        x = self.conv1(data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.sigmoid(x)

        return x


def ResNet34(num_classes=55):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50_Basic(num_classes=55):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101_Basic(num_classes=55):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=55):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def test_ResNet():
    net = ResNet50().cuda()
    print( summary(net, input_size=(8, 5000)) )


if __name__ == '__main__':
    test_ResNet()
