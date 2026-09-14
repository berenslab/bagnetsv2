"""Adapted from: https://github.com/pytorch/examples/blob/main/imagenet/main.py"""

import time
from collections import defaultdict

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import get_model

from bagnetsv2 import bagnetsv2 as bagnets
from bagnetsv2 import utils


def get_dataloader(cfg):
    image_size = (cfg.train.image_size, cfg.train.image_size)
    transform = utils.get_augmentations(
        image_size, normalization=utils.IMAGENET_NORMALIZATION, imagenet=True
    )

    dataset_train, dataset_val, n_classes = utils.load_train_dataset(
        cfg.dataset.dir, name=cfg.dataset.name, transform=transform, seed=cfg.seed
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader_train, dataloader_val, n_classes


def get_train_objs(cfg, n_classes, checkpoint_file, wandb_kwargs):
    """Load model, optimizer, scheduler and loss function, resuming from a checkpoint if one exists.

    Returns:
        tuple: model, optimizer, scheduler, loss_fn, start_epoch, checkpoint (the loaded
            checkpoint dict, or None if training starts from scratch).
    """
    if 'bagnet' in cfg.model.variant:
        model = bagnets.get_bagnet(
            cfg.model.variant, weights=None, num_classes=n_classes
        )
    else:
        model = get_model(cfg.model.variant, weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    optimizer = utils.get_optimizer(
        cfg.optim.name, model.parameters(), cfg.optim.lr, cfg.optim.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    # Load checkpoint if it exists
    start_epoch = 0
    stats = defaultdict(list)
    checkpoint = None
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        stats = defaultdict(list, checkpoint['stats'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # Resume logging to the same id
        wandb_kwargs['id'] = checkpoint['wandb_id']
        wandb_kwargs['resume'] = 'must'

    return model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs


def train(model, dataloader, loss_fn, optimizer, scaler, device):
    model.to(device)
    model.train()
    epoch_loss, epoch_acc1, epoch_acc5 = 0.0, 0.0, 0.0
    for b, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels.type(torch.int64))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = utils.accuracy(outputs, labels, topk=(1, 5))
        epoch_loss += loss.item()
        epoch_acc1 += acc1.item()
        epoch_acc5 += acc5.item()

    n_batches = len(dataloader)
    return epoch_loss / n_batches, epoch_acc1 / n_batches, epoch_acc5 / n_batches


def validate(model, dataloader, loss_fn, device):
    model.to(device)
    model.eval()
    val_loss, val_acc1, val_acc5 = 0.0, 0.0, 0.0
    for b, (img, labels) in enumerate(dataloader):
        img, labels = img.to(device), labels.to(device)

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(img)
            loss = loss_fn(outputs, labels.type(torch.int64))

        acc1, acc5 = utils.accuracy(outputs, labels, topk=(1, 5))
        val_loss += loss.item()
        val_acc1 += acc1.item()
        val_acc5 += acc5.item()

    n_batches = len(dataloader)
    return val_loss / n_batches, val_acc1 / n_batches, val_acc5 / n_batches


@hydra.main(version_base=None, config_path='../../configs', config_name='default')
def main(cfg: DictConfig):
    start = time.perf_counter()
    print(cfg.experiment_name)
    utils.validate_config(cfg)
    utils.set_seed(cfg.seed)

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_train, dataloader_val, n_classes = get_dataloader(cfg)
    wandb_kwargs, checkpoint_file = utils.init_experiment(cfg)
    model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs = (
        get_train_objs(cfg, n_classes, checkpoint_file, wandb_kwargs)
    )

    # Compile model
    uncompiled_model = model
    model = torch.compile(uncompiled_model)  # Set drop_last=True in the dataloader

    ###################### TRAINING LOOP #########################
    with wandb.init(**wandb_kwargs) as run:
        # Define metrics: x-axis, other metrics
        run.define_metric('epoch')
        run.define_metric('loss/*', step_metric='epoch', summary='min')
        run.define_metric('acc1/*', step_metric='epoch', summary='max')
        run.define_metric('acc5/*', step_metric='epoch', summary='max')

        early_stopping = utils.EarlyStopping(
            patience=cfg.stop.patience,
            min_delta=cfg.stop.min_delta,
            checkpoint_file=checkpoint_file,
            verbose=True,
        )
        scaler = torch.amp.GradScaler()
        for epoch in range(start_epoch, cfg.train.epochs):
            start_time = time.perf_counter()

            # Training
            train_loss, train_acc1, train_acc5 = train(
                model, dataloader_train, loss_fn, optimizer, scaler, cfg.device
            )
            stats['loss_train'].append(train_loss)
            stats['acc1_train'].append(train_acc1)
            stats['acc5_train'].append(train_acc5)

            # Validation
            val_loss, val_acc1, val_acc5 = validate(
                model, dataloader_val, loss_fn, cfg.device
            )
            stats['loss_val'].append(val_loss)
            stats['acc1_val'].append(val_acc1)
            stats['acc5_val'].append(val_acc5)

            scheduler.step()

            run.log(
                {
                    'loss/train': train_loss,
                    'loss/val': val_loss,
                    'acc1/train': train_acc1,
                    'acc1/val': val_acc1,
                    'acc5/train': train_acc5,
                    'acc5/val': val_acc5,
                },
                step=epoch,
            )

            # Early stopping
            checkpoint = {
                'state_dict': uncompiled_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'stats': stats,
                'wandb_id': run.id,
            }
            early_stopping(val_loss, checkpoint)
            if early_stopping.stop_training:
                break

            end_time = time.perf_counter()
            print(
                f'Epoch {epoch + 1}, train loss {train_loss:.3f}, val loss {val_loss:.3f}, {end_time - start_time:.1f} s'
            )

    print(f'Total training time: {time.perf_counter() - start:.1f} s')

    ###################### EVALUATION #########################
    model, _, _, _, _, _, _ = get_train_objs(cfg, n_classes, checkpoint_file, wandb_kwargs)

    val_loss, val_acc1, val_acc5 = validate(model, dataloader_val, loss_fn, cfg.device)
    print(f'Top 1 accuracy on validation set: {val_acc1:.2f}')
    print(f'Top 5 accuracy on validation set: {val_acc5:.2f}')


if __name__ == '__main__':
    main()
