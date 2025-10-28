'''BagNet implementation adapted from: https://github.com/wielandbrendel/bag-of-local-features-models/blob/master/bagnets/pytorchnet.py'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils import model_zoo
import skimage

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

__all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
            'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
            'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
            'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
                            }


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
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
        
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]
        
        out += residual
        out = self.relu(out)

        return out


class BagNet(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], num_classes=1000, avg_pool=True):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0,2,3,1)
            x = self.fc(x)

        return x
    

def get_bagnet(name, weights, strides=[2, 2, 2, 1], **kwargs):
    """
    Get bagnet33, bagnet17 or bagnet9.
    
    Parameters
    ----------
    name : string
        This should be one of the BagNets.
    weights : string
        Can be DEFAULT or None, DEFAULT will load the weights pretrained 
        on imagenet, anything else will return a model with randomly 
        initialized weights.
    """
    if name == 'bagnet33':
        model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,1], **kwargs)
    elif name == 'bagnet17':
        model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,0], **kwargs)
    elif name == 'bagnet9':
        model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,0,0], **kwargs)
    else:
        raise Exception('Bagnet not supported, try bagnet9, bagnet17 or bagnet33')
    
    # To mimic tochvision get_model
    if weights == 'DEFAULT':
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls[name], map_location=torch.device('cpu'), weights_only=True))

    return model


def generate_heatmap(model, image, target, patchsize, device):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.
    
    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Torch tensor of shape [3, X, X]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.
    
    """
    # pad image with zeros
    _, h, w = image.shape
    image = transforms.functional.pad(image, padding=patchsize//2, fill=0, padding_mode='constant').unsqueeze(0)

    # extract patches: unfold(dim, size, step)
    patches = image.permute(0, 2, 3, 1)
    patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
    patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

    # compute logits for each patch
    model.to(device)
    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch_patches in torch.split(patches, 512):
            batch_patches = batch_patches.to(device)
            logits = model(batch_patches)
            logits_list.extend(logits[:, target].tolist())

    logits = np.array(logits_list)
    return logits.reshape((h, w))
    

def plot_heatmap(heatmap, original, fig, ax, cmap='RdBu_r', percentile=99, alpha=0.25, mask=None):
    """
    Plots the heatmap on top of the original image 
    (which is shown by most important edges).
    
    Parameters
    ----------
    heatmap : Numpy Array of shape [Y, Y]
        Heatmap to visualise.
    original : Torch tensor of shape [3, X, X]
        Original image for which the heatmap was computed.
    ax : Matplotlib axis
        Axis onto which the heatmap should be plotted.
    cmap : Matplotlib color map
        Color map for the visualisation of the heatmaps (default: RdBu_r)
    percentile : float in [0, 100] (default: 99)
        Extreme values outside of the percentile range are clipped.
        This avoids that a single outlier dominates the whole heatmap.
    alpha : float in [0, 1]
        Opacity of the overlay image.
    mask : Numpy Array of shape [X, X]
        Mask to set nan values to the heatmap outside the area of interest.
    """

    if original is not None:
        # Image from tensor to numpy array and grayscale
        original = np.transpose(original.numpy(), (1, 2, 0)).mean(axis=-1) 

        # Compute edges (to overlay to heatmap)
        edges = skimage.feature.canny(original, sigma=1)
        overlay = np.where(edges, 0, np.nan)

    extent = (-0.5, original.shape[0] - 0.5, original.shape[1] - 0.5, -0.5)
    
    perc = np.nanpercentile(np.abs(heatmap), percentile)
    # print(np.nanmin(heatmap), np.nanmax(heatmap), perc)
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(alpha=0)

    hm = ax.imshow(heatmap, interpolation='bilinear', cmap=cmap, vmin=-perc, vmax=perc, extent=extent)
    fig.colorbar(hm, ax=ax, shrink=0.2)

    if mask is not None:
        ax.imshow(mask, interpolation='none', cmap='Greys', extent=extent)

    if overlay is not None:
        cmap_original = plt.get_cmap('Greys_r')
        cmap_original.set_bad(alpha=0)
        ax.imshow(overlay, interpolation='none', cmap=cmap_original, alpha=alpha, extent=extent)
    ax.axis('off')
        