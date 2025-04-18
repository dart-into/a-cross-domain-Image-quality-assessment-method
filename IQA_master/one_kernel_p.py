import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from utils import load_state_dict_from_url
from collections import OrderedDict

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.k1 = nn.Parameter(torch.tensor(0.3333, dtype=torch.float32))
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(self.k1*x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer( num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)



class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        #self.c = nn.Conv2d(2304, 256, kernel_size=1, stride=1, bias=False)
        # First convolution
        #self.k1 = nn.Parameter(torch.tensor(0.333, dtype=torch.float32))
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        block1 = _DenseBlock(num_layers=6, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % (1), block1)
        num_features = num_features + 6 * growth_rate
        
        
        trans1 = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
        self.features.add_module('transition%d' % (1), trans1)
        num_features = num_features // 2
     
        
        block2 = _DenseBlock(num_layers=12, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % (2), block2)
        num_features = num_features + 12 * growth_rate
        
        
        trans2 = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
        self.features.add_module('transition%d' % (2), trans2)
        num_features = num_features // 2
        
    
        block3 = _DenseBlock(num_layers=24, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % (3), block3)
        num_features = num_features + 24 * growth_rate
        
        trans3 = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
        self.features.add_module('transition%d' % (3), trans3)
        num_features = num_features // 2
        
        block4 = _DenseBlock(num_layers=16, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % (4), block4)
        num_features = num_features + 16 * growth_rate

        #self.features.con_share = nn.Conv2d(512, 16, kernel_size=1, stride=1, bias=False)
        #self.features.con_x = nn.Conv2d(128, 112, kernel_size=1, stride=1, bias=False)

        '''
        #Final merge block
        self.merge_net = _DenseBlock(num_layers=8, num_input_features=num_features*3,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
        num_features = 1792
        # Final batch norm
        self.merge_net.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)

        # Linear layer
        self.classifier1 = nn.Linear(1792, 1024)
        self.classifier2 = nn.BatchNorm1d(1024)
        #self.classifier4 = nn.PReLU()
        self.classifier3 = nn.Dropout(p=0.5)
        self.classifier4 = nn.Linear(1024, 1000)
        self.classifier5 = nn.BatchNorm1d(1000)
        #self.classifier8 = nn.PReLU()
        self.classifier6 = nn.Dropout(p=0.5)
        '''

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_56):

        #the size of share_feature: 512*28*28
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        
        x = self.features.denseblock1(x_56) #256,56,56
        x_56 = x
        #print('1', x.shape)
        x = self.features.transition1(x) #128,28,28
        x_28 = x
        #print('2', x.shape)
        
        #share_feature = self.features.con_share(share_feature)
        #x = self.features.con_x(x)
        
        #x = torch.cat((x, share_feature), dim=1)
        
        #x = torch.cat((x, x3, x5), dim=1)
        x = self.features.denseblock2(x) #512,28,28
        #x_28 = x
        #print('3', x.shape)
        x = self.features.transition2(x) #256,14,14
        #print('4', x.shape)
        
        #x = torch.cat((x, x3, x5), dim=1)
        
        #x = self.c(x)
        
        out = self.features.denseblock3(x)
        
        
        return out, x_28

'''
def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True)
    #state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
'''

def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
              **kwargs):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model



def densenetnew(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet121', 32, (6, 12, 24), 64, pretrained, progress,
                     **kwargs)

'''
def densenet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)
'''

'''
[docs]def densenet161(pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                     **kwargs)



[docs]def densenet169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                     **kwargs)



[docs]def densenet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                     **kwargs)
'''