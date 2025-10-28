import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.models import get_model

import bagnetsv2 as bagnets # Change to import bagnets to see the results of the original model
import pretrain_imagenet as pt
import utils


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation on ImageNet.')
    parser.add_argument('--backbone', default='bagnet33', type=str, help='backbone model', choices=['resnet18', 'resnet50', 'bagnet33', 'bagnet17', 'bagnet9'])
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to train on', choices=['imagenet', 'imagenette'])
    parser.add_argument('--checkpoint', default='checkpoints/bagnet33_imagenet_pretrained.pt', type=str, help='filename of the checkpoint to evaluate')
    parser.add_argument('--imagesize', default=224, type=int, help='image size, only square images are supported')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size for training')
    parser.add_argument('--numworkers', default=4, type=int, help='number of subprocesses to use for dataloading')
    parser.add_argument('--device', default='cuda:0', type=str, help='device in which training will take place')

    args = parser.parse_args()
    args.imagesize = (args.imagesize, args.imagesize)
    return args


def get_dataloader(args):
    transform = utils.get_augmentations(args.imagesize, normalization=utils.IMAGENET_NORMALIZATION, imagenet=True)['test']
    
    if args.dataset == 'imagenet':
        dataset_test = datasets.ImageNet(utils.IMAGENET_DIR, split='val', transform=transform)
        n_classes = 1000
    else:
        dataset_test = datasets.Imagenette('datasets', split='val', transform=transform, size='320px', download=True)
        n_classes = 10

    dataloader_test = DataLoader(dataset_test, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=True, drop_last=True)

    return dataloader_test, n_classes


def get_model(args, n_classes):
    if 'bagnet' in args.backbone:
        model = bagnets.get_bagnet(args.backbone, weights=None, num_classes=n_classes)
    else:
        model = get_model(args.backbone, weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    args = get_args()
    start = time.perf_counter()

    dataloader_test, n_classes = get_dataloader(args)
    model = get_model(args, n_classes)

    # Check for dead convolutional layers
    dead_layer_count = 0
    for name, parameters in model.named_parameters():
        if 'conv' in name:
            max_weight = parameters.flatten().abs().max()

            if max_weight <= 1e-4:
                dead_layer_count += 1

    print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

    # Accuracy
    preds, probs, targets = pt.predict(model, dataloader_test, args.device)

    acc = utils.accuracy(torch.from_numpy(probs), torch.from_numpy(targets), (1, 5))
    print(f'Top 1 accuracy on validation set: {acc[0].item():.2f}')
    print(f'Top 5 accuracy on validation set: {acc[1].item():.2f}')

    print(f'Total testing time: {time.perf_counter() - start:.1f} s')
