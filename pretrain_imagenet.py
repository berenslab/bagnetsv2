'''Adapted from: https://github.com/pytorch/examples/blob/main/imagenet/main.py'''
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
    parser.add_argument('--device', default='cuda:0', type=str, help='device in which training will take place')

    args = parser.parse_args()
    args.imagesize = (args.imagesize, args.imagesize)
    return args


def get_dataloader(args):
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

    dataloader_train = DataLoader(dataset_train, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batchsize, shuffle=False, num_workers=args.numworkers, pin_memory=True, drop_last=True)

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
    model.to(device)
    model.train()
    epoch_loss = 0.0
    for b, (imgs, labels) in enumerate(dataloader):
        # print(f'Batch {b + 1} out of {len(dataloader_train)}...')
        # start_time = time.time()
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels.type(torch.int64))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        # print('Time per batch = ', time.time() - start_time)

    epoch_loss = epoch_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    model.to(device)
    model.eval()
    val_loss = 0.0
    for b, (img, labels) in enumerate(dataloader):
        img, labels = img.to(device), labels.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(img)

        loss = loss_fn(outputs, labels.type(torch.int64))
        val_loss += loss.item()

    val_loss = val_loss / len(dataloader)
    return val_loss


def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    preds, probs, targets, embeddings = [], [], [], []
    for b, (img, labels) in enumerate(tqdm(dataloader)):            
        img, labels = img.to(device), labels.to(device)

        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y = model(img)
                pred = torch.max(y, dim=1)
                p = F.softmax(y, dim=1)
    
        preds.extend(pred.indices.tolist())
        probs.extend(p.tolist())
        targets.extend(labels.tolist())

    preds = np.array(preds)
    probs = np.array(probs)
    targets = np.array(targets)

    return preds, probs, targets

    
if __name__ == '__main__':
    args = get_args()
    start = time.time()

    ##################### TRAINING PARAMS ######################
    experiment_name = f'{args.backbone}_{args.dataset}'
    print(experiment_name)

    # Where results will be saved
    project_dir = Path.cwd()
    checkpoints_dir = project_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoints_dir.joinpath(f'{experiment_name}.pt')

    plots_dir = project_dir.joinpath('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plots_dir.joinpath(f'{experiment_name}.png')
    
    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_train, dataloader_val, n_classes = get_dataloader(args)
    model, optimizer, scheduler, loss_fn, start_epoch = get_train_objs(args, n_classes, checkpoint_file)
    
    # Compile model
    uncompiled_model = model
    model = torch.compile(uncompiled_model) # Set drop_last=True in the dataloader

    ###################### TRAINING LOOP #########################
    early_stopping = utils.EarlyStopping(patience=10, min_delta=1e-4, checkpoint_file=checkpoint_file, verbose=True)
    scaler = torch.amp.GradScaler()
    loss_curves = {'loss_train': [], 'loss_val': []}
    for epoch in range(args.epochs):
        start_time = time.time()

        # Training
        train_loss = train(model, dataloader_train, loss_fn, optimizer, scaler, args.device)
        loss_curves['loss_train'].append(train_loss)       

        # Validation
        val_loss = validate(model, dataloader_val, loss_fn, args.device)
        loss_curves['loss_val'].append(val_loss)
        scheduler.step()

        # Early stopping
        checkpoint = {'state_dict': model.module.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict(), 
                        'epoch': epoch + 1} | loss_curves
        early_stopping(val_loss, uncompiled_model)
        if early_stopping.stop_training:
            break
        
        end_time = time.time()
        print(f"Epoch {epoch + 1}, train loss {train_loss:.3f}, val loss {val_loss:.3f}, {end_time - start_time:.1f} s")

    ###################### TRAINING CURVE #########################
    loss_train = checkpoint['loss_train']
    loss_val = checkpoint['loss_val']

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(loss_train)), loss_train, '-o')
    ax.plot(range(len(loss_val)), loss_val, '-o')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Loss')
    ax.legend(['Train', 'Val'])
    plt.tight_layout()
    fig.savefig(plot_file)

    print(f'Total training time: {time.time() - start:.1f} s')

    ###################### EVALUATION #########################
    model, _, _, _, _ = get_train_objs(args, n_classes, checkpoint_file)

    preds, probs, targets = predict(model, dataloader_val, args.device)
    acc = utils.accuracy(torch.from_numpy(probs), torch.from_numpy(targets))
    print(f'Top 1 accuracy on validation set: {acc[0].item():.2f}')
    