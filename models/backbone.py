import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


class ResNet(nn.Module):

    def __init__(self, block, layers,
                 inplanes=64,
                 num_input_channels=3,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 device="cuda:0",
                 **kwargs):
        super(ResNet, self).__init__()

        self.inplanes = inplanes

        self.deconv_with_bias = deconv_with_bias
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_filters = num_deconv_filters
        self.num_deconv_kernels = num_deconv_kernels
        
        self.conv1 = nn.Conv2d(num_input_channels, self.inplanes, 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.deconv_layers = self._make_deconv_layer(
            self.num_deconv_layers,
            self.num_deconv_filters,
            self.num_deconv_kernels
        )
        self.to(device)
            
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        # assert num_layers == len(num_filters), \
        #     'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        # assert num_layers == len(num_kernels), \
        #     'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.deconv_layers(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def build_backbone(cfg):
    arch = cfg.MODEL.BACKBONE.ARCHITECTURE
    weight_pth = cfg.MODEL.BACKBONE.INIT_WEIGHT
    
    if arch == "resnet18":
        blcok = resnet.BasicBlock
        layers = (2, 2, 2, 2)
    elif arch == "resnet34":
        block = resnet.BasicBlock
        layers = (3, 4, 6, 3)
    elif arch == "resnet50":
        block = resnet.Bottleneck
        layers = (3, 4, 6, 3)
    elif arch == "resnet101":
        block = resnet.Bottleneck
        layers = (3, 4, 23, 3)
    elif arch == "resnet152":
        block = resnet.Bottleneck
        layers = (3, 8, 36, 3)
    else:
        NotImplementedError, "Backbone name {} is not implemented".format(arch)

    backbone = ResNet(block, layers,
                      inplanes=cfg.MODEL.BACKBONE.INPLANES,
                      deconv_with_bias=False,
                      num_deconv_layers=cfg.MODEL.BACKBONE.DECONV_LAYERS,
                      num_deconv_filters=cfg.MODEL.BACKBONE.DECONV_FILTERS,
                      num_deconv_kernels=cfg.MODEL.BACKBONE.DECONV_KERNELS,
                      device=cfg.DEVICE)

    if weight_pth is not "":
        state_dict = torch.load(cfg.MODEL.BACKBONE.INIT_WEIGHT)
        if cfg.MODEL.BACKBONE.INIT_WEIGHT.split('/')[-1] == 'LT_backbone.pth':
            state_dict = {k[7:]: v for k, v in state_dict.items() if k[7:] in backbone.state_dict().keys()}
        try:
            backbone.load_state_dict(state_dict)
        except:
            AssertionError, "Pretrained weight {} does not match with model architecture {}".format(weight_pth, arch)

    return backbone