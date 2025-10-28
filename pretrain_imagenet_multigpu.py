'''Adapted from: https://github.com/pytorch/examples/blob/main/imagenet/main.py
To run use torchrun --standalone --nproc_per_node=2 pretrain_imagenet_multigpu.py
'''

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.models import get_model

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import bagnetsv2 as bagnets
import utils


def get_args():
    parser = argparse.ArgumentParser(description='Supervised pre-training on ImageNet.')
    parser.add_argument('--backbone', default='bagnet33', type=str, help='backbone model', choices=['resnet18', 'resnet50', 'bagnet33', 'bagnet17', 'bagnet9'])
    parser.add_argument('--dataset', default='imagenette', type=str, help='dataset to train on', choices=['imagenet', 'imagenette'])
    parser.add_argument('--imagesize', default=224, type=int, help='image size, only square images are supported')
    parser.add_argument('--batchsize', default=256, type=int, help='batch size for training')
    parser.add_argument('--epochs', default=90, type=int, help='number of epochs for training')
    parser.add_argument('--numworkers', default=4, type=int, help='number of subprocesses to use for dataloading')

    args = parser.parse_args()
    args.imagesize = (args.imagesize, args.imagesize)
    return args


def ddp_setup():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK'])) # rank = unique identifier of each process
    dist.init_process_group(backend='nccl')


def main_worker(checkpoint_file, args):
    ddp_setup()
    device = int(os.environ['LOCAL_RANK']) # Provided by torchrun
    world_size = int(os.environ['LOCAL_WORLD_SIZE'])
    start_time_train = time.time()

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_train, dataloader_val, n_classes = get_dataloader(args, world_size)
    model, optimizer, scheduler, loss_fn, start_epoch = get_train_objs(args, n_classes, checkpoint_file)

    # Prepare model
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
    # model = torch.compile(model) # Set drop_last=True in the dataloader, creates warning with for grad strides and makes it slower
    model = DDP(model, device_ids=[device])
    
    # Train
    early_stopping = utils.EarlyStopping(patience=10, min_delta=1e-4, checkpoint_file=checkpoint_file, verbose=True)
    scaler = torch.amp.GradScaler()
    loss_curves = {'loss_train': [], 'loss_val': []}
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Training
        dataloader_train.sampler.set_epoch(epoch)
        train_loss = train(model, dataloader_train, loss_fn, optimizer, scaler, device)
        train_loss = reduce(train_loss, device).item() # Get training loss across all processes    

        # Validation
        val_loss = validate(model, dataloader_val, loss_fn, device)
        val_loss = reduce(val_loss, device).item() # Get training loss across all processes      
        
        scheduler.step()

        # Logging and early stopping
        if device == 0:
            loss_curves['loss_train'].append(train_loss)
            loss_curves['loss_val'].append(val_loss)

            checkpoint = {'state_dict': model.module.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict(), 
                        'epoch': epoch + 1} | loss_curves
            early_stopping(val_loss, checkpoint)
            if early_stopping.stop_training:
                break

            print(f"Epoch {epoch + 1}, train loss {train_loss:.3f}, val loss {val_loss:.3f}, {time.time() - start_time:.1f} s")
        dist.barrier() # To ensure other ranks wait for model to be saved to disk before continuing training

    if device == 0:
        print(f'Total training time: {time.time() - start_time_train:.1f} s')
    dist.destroy_process_group()


def get_dataloader(args, world_size):
    # Set batch size per gpu, effective batch size is equal to args.batchsize
    batch_size = args.batchsize // world_size

    transform = utils.get_augmentations(args.imagesize, normalization=utils.IMAGENET_NORMALIZATION, imagenet=True)
    
    if args.dataset == 'imagenet':
        dataset_train = datasets.ImageNet(utils.IMAGENET_DIR, split='train', transform=transform['train'])
        dataset_val = datasets.ImageNet(utils.IMAGENET_DIR, split='train', transform=transform['test'])
        n_classes = 1000
    else:
        dataset_train = datasets.Imagenette('datasets', split='train', transform=transform['train'], size='320px', download=True)
        dataset_val = datasets.Imagenette('datasets', split='train', transform=transform['test'], size='320px', download=True)
        n_classes = 10
    
    # Split training dataset into training and validation set with different transforms
    dataset_train, dataset_val = utils.split_imagenet_train_val(dataset_train, dataset_val, val_size=0.05)

    sampler_train = DistributedSampler(dataset_train, shuffle=True, drop_last=True) 
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=args.numworkers, pin_memory=True, sampler=sampler_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=args.numworkers, pin_memory=True, sampler=sampler_val)

    return dataloader_train, dataloader_val, n_classes


def get_train_objs(args, n_classes, checkpoint_file):
    # Load model, optimizer, scheduler and loss function
    if 'bagnet' in args.backbone:
        model = bagnets.get_bagnet(args.backbone, weights=None, num_classes=n_classes)
    else:
        model = get_model(args.backbone, weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    optimizer = torch.optim.SGD(model.parameters(), 0.01 * args.batchsize / 256, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    # Load checkpoint if it exists
    start_epoch = 0
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, loss_fn, start_epoch


def train(model, dataloader, loss_fn, optimizer, scaler, device):
    model.train()
    epoch_loss = 0.0
    for b, (imgs, labels) in enumerate(dataloader):
        # print(f'Batch {b + 1} out of {len(dataloader)}...')
        # start = time.time()
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels.type(torch.int64))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        # print('time per batch', time.time() - start, 's, for batch size of', imgs.shape[0])

    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    for b, (img, labels) in enumerate(dataloader):
        # print(f'Batch {b + 1} out of {len(dataloader)}...')
        # start = time.time()
        img, labels = img.to(device), labels.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(img)

        loss = loss_fn(outputs, labels.type(torch.int64))
        val_loss += loss.item()
        # print('time per batch', time.time() - start, 's, for batch size of', img.shape[0])

    val_loss = val_loss / len(dataloader)
    return val_loss


def reduce(value, device):
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value

    
if __name__ == '__main__':
    args = get_args()

    ##################### TRAINING PARAMS ######################
    experiment_name = f'{args.backbone}_{args.dataset}'
    print(experiment_name)

    # Where results will be saved
    project_dir = Path.cwd()
    checkpoints_dir = project_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoints_dir.joinpath(f'{experiment_name}.pt')

    ###################### TRAINING LOOP #########################
    main_worker(checkpoint_file=checkpoint_file, args=args)
