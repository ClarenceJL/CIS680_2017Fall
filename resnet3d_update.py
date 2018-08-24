from layer.conv_decomposed import *

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200', 'get_fine_tuning_parameters',
    'resnet18_2plus1d', 'resnet34_2plus1d'
]

""" 
    Models pre-trained on kinetics:
    https://drive.google.com/drive/folders/14KRBqT8ySfPtFSuLsFS2U4I-ihTDs0Y9
"""
model_paths = {
    'resnet50': 'E:/WorkSpace/pretrained_model/resnet-50-kinetics.pth'
}
model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv2plus1(inplanes, planes, k_size=3, stride=1, padding=0,reversed=False, bias=False, middleplanes=None):
    if not isinstance(k_size, (tuple, list)):
        k_size = (k_size, k_size, k_size)
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding, padding)
    # middleplanes = planes
    # if we follow the exact implementation for R(2+1)D, we would have:
    if middleplanes is None:
        middleplanes = k_size[0] * inplanes * planes * k_size[1] * k_size[2]
        if reversed:
            middleplanes /= (inplanes * k_size[0] + planes * k_size[1] * k_size[2])
        else:
            middleplanes /= (inplanes * k_size[1] * k_size[2] + k_size[0] * planes)

    middleplanes = int(middleplanes)
    if reversed:
        return nn.Sequential(
            nn.Conv3d(inplanes, middleplanes,
                      kernel_size=(k_size[0], 1, 1),
                      stride=(stride[0], 1, 1),
                      padding=(padding[0], 0, 0),
                      bias=bias),
            nn.BatchNorm3d(middleplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(middleplanes, planes,
                      kernel_size=(1, k_size[1], k_size[2]),
                      stride=(1, stride[1], stride[2]),
                      padding=(0, padding[1], padding[2]),
                      bias=bias))
    else:
        return nn.Sequential(
            nn.Conv3d(inplanes, middleplanes,
                      kernel_size=(1, k_size[1], k_size[2]),
                      stride=(1, stride[1], stride[2]),
                      padding=(0, padding[1], padding[2]),
                      bias=bias),
            nn.BatchNorm3d(middleplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(middleplanes, planes,
                      kernel_size=(k_size[0], 1, 1),
                      stride=(stride[0], 1, 1),
                      padding=(padding[0], 0, 0),
                      bias=bias))
                      
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, decomposed=False, reversed=False):
        super(BasicBlock, self).__init__()
        if decomposed:
            self.conv1 = conv2plus1(inplanes, planes, stride=stride, padding=1, reversed=reversed, bias=False)
        else:
            self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        if decomposed:
            self.conv2 = conv2plus1(planes, planes, padding=1, reversed=reversed, bias=False)  # for the rest conv, set stride = 1
        else:
            self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, eps=1e-3)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, decomposed=False, reversed=False):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.decomposed = decomposed
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        if decomposed:
            self.conv2 = conv2plus1(planes, planes, k_size=3, stride=stride, padding=1, bias=False, reversed=reversed)
        else:
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 num_classes=400,
                 last_conv_stride=2,
                 decomposed=False,
                 reversed=False):
        self.decomposed = decomposed
        self.reversed = reversed
        self.inplanes = 64
        super(ResNet, self).__init__()
        if self.decomposed:
            self.conv1 = conv2plus1(3, 64, (3, 7, 7), (1, 2, 2), 3, reversed=reversed, bias=False)
        else:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=last_conv_stride)
        # last_duration = int(math.ceil(sample_duration / 16))
        # last_size = int(math.ceil(sample_size / 32))
        # self.avgpool = nn.AvgPool3d(
        #     (last_duration, last_size, last_size), stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),  # add a bottleneck
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, decomposed=self.decomposed, reversed=self.reversed))
        self.inplanes = planes * block.expansion  # change number of channels as network goes deeper
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, decomposed=self.decomposed, reversed=self.reversed))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.dropout(x, training=self.training)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    # def get_modules(self):
    #     return self.modules()


def get_fine_tuning_parameters(model, ft_begin_index, only_freeze_BN):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            if only_freeze_BN:
                if 'bn' in k:
                    print(k)
                    parameters.append({'params': v, 'lr': 0.0})
                else:
                    parameters.append({'params': v})
            else:
                parameters.append({'params': v, 'lr': 0.0})

    return parameters


def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  for key, value in state_dict.items():
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print('load pretrained model ... ')
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])), strict=False)
    return model


def resnet34(pretrained = False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(torch.load(model_paths['resnet50'], map_location='cpu')), strict=False)
        # model.load_state_dict(remove_fc(torch.load(model_paths['resnet50'])), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet101'])), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet152'])), strict=False)
    return model


def resnet200(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet200'])), strict=False)
    return model


##########################################################################################################


def resnet18_2plus1d(pretrained=False, reversed=False, **kwargs):
    """Constructs a ResNet-18 model with R(2+1)D block.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], decomposed=True, reversed=reversed, **kwargs)
    if pretrained:
        print('load pretrained model ... ')
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18_2plus1'])), strict=False)
    return model


def resnet34_2plus1d(pretrained=False, reversed=False, **kwargs):
    """Constructs a ResNet-34 model with R(2+1)D block.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], decomposed=True, reversed=reversed, **kwargs)
    if pretrained:
        print('load pretrained model ... ')
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34_2plus1'])), strict=False)
    return model


def resnet50_2plus1d(pretrained=False, reversed=False, **kwargs):
    """Constructs a ResNet-50 model with R(2+1)D block.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], decomposed=True, reversed=reversed, **kwargs)
    if pretrained:
        print('load pretrained model ...')
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50_2plus1'])), strict=False)
    return model


def resnet101_2plus1d(pretrained=False, reversed=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], reversed=reversed, **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet101_2plus1'])), strict=False)
    return model


def resnet152_2plus1d(pretrained=False, reversed=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], reversed=reversed, **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet152_2plus1'])), strict=False)
    return model


def resnet200_2plus1d(pretrained=False, reversed=False, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], reversed=reversed, **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet200_2plus1'])), strict=False)
    return model


##########################################################################################################


def resnet18_mc(pretrained=False, **kwargs):
    NotImplementedError


def resnet18_rmc(pretrained=False, **kwargs):
    NotImplementedError


def resnet34_mc(pretrained=False, **kwargs):
    NotImplementedError


def resnet34_rmc(pretrained=False, **kwargs):
    NotImplementedError
