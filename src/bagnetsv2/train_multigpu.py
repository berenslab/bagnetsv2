"""Adapted from: https://github.com/pytorch/examples/blob/main/imagenet/main.py
To run use torchrun --standalone --nproc_per_node=2 -m bagnetsv2.train_multigpu
"""

import time
from collections import defaultdict

import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import get_model

from bagnetsv2 import bagnetsv2 as bagnets
from bagnetsv2 import utils


def get_dataloader(cfg, world_size):
    # Set batch size per gpu, effective batch size is equal to cfg.train.batch_size
    batch_size = cfg.train.batch_size // world_size
    image_size = (cfg.train.image_size, cfg.train.image_size)

    transform = utils.get_augmentations(
        image_size, normalization=utils.IMAGENET_NORMALIZATION, imagenet=True
    )

    dataset_train, dataset_val, n_classes = utils.load_train_dataset(
        cfg.dataset.dir, name=cfg.dataset.name, transform=transform, seed=cfg.seed
    )

    sampler_train = DistributedSampler(dataset_train, shuffle=True, drop_last=True)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        sampler=sampler_train,
    )
    sampler_val = DistributedSampler(dataset_val, shuffle=False, drop_last=True)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        sampler=sampler_val,
    )

    return dataloader_train, dataloader_val, n_classes


def get_train_objs(cfg, n_classes, checkpoint_file, wandb_kwargs):
    """Load model, optimizer, scheduler and loss function, resuming from a checkpoint if one exists.

    Returns:
        tuple: model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs.
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
    model.train()
    epoch_loss, epoch_acc1, epoch_acc5 = 0.0, 0.0, 0.0
    for b, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)
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
    model.eval()
    val_loss, val_acc1, val_acc5 = 0.0, 0.0, 0.0
    for b, (img, labels) in enumerate(dataloader):
        img = img.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)

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
    print(cfg.experiment_name)
    utils.validate_config(cfg)
    utils.set_seed(cfg.seed)

    # DDP setup
    rank, local_rank, world_size = utils.ddp_setup()
    wandb_kwargs, checkpoint_file = utils.init_experiment(cfg, rank=rank)

    start_time_train = time.perf_counter()

    # Get dataloader, model, optimizer, scheduler and loss function
    dataloader_train, dataloader_val, n_classes = get_dataloader(cfg, world_size)
    model, optimizer, scheduler, loss_fn, start_epoch, stats, wandb_kwargs = (
        get_train_objs(cfg, n_classes, checkpoint_file, wandb_kwargs)
    )

    run = None
    if rank == 0:
        run = wandb.init(**wandb_kwargs)

    # Prepare model
    model.to(local_rank, memory_format=torch.channels_last)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.compile(model) # Set drop_last=True in the dataloader, slower
    model = DDP(model, device_ids=[local_rank])

    # Train
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
        dataloader_train.sampler.set_epoch(epoch)
        train_loss, train_acc1, train_acc5 = train(
            model, dataloader_train, loss_fn, optimizer, scaler, local_rank
        )
        train_loss, train_acc1, train_acc5 = utils.reduce(
            [train_loss, train_acc1, train_acc5], local_rank
        ).tolist()

        # Validation
        val_loss, val_acc1, val_acc5 = validate(
            model, dataloader_val, loss_fn, local_rank
        )
        val_loss, val_acc1, val_acc5 = utils.reduce(
            [val_loss, val_acc1, val_acc5], local_rank
        ).tolist()

        scheduler.step()

        # Logging and early stopping
        if rank == 0:
            stats['loss_train'].append(train_loss)
            stats['loss_val'].append(val_loss)
            stats['acc1_train'].append(train_acc1)
            stats['acc1_val'].append(val_acc1)
            stats['acc5_train'].append(train_acc5)
            stats['acc5_val'].append(val_acc5)
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

            checkpoint = {
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'stats': stats,
                'wandb_id': run.id,
            }
            early_stopping(val_loss, checkpoint)

        # All ranks wait for rank 0 to finish saving the checkpoint
        dist.barrier()

        # Broadcast stop decision from rank 0 so all ranks break together
        stop_signal = torch.tensor(
            int(rank == 0 and early_stopping.stop_training),
            device=local_rank,
        )
        dist.broadcast(stop_signal, src=0)

        if rank == 0:
            end_time = time.perf_counter()
            print(
                f'Epoch {epoch + 1}, train loss {train_loss:.3f}, val loss {val_loss:.3f}, {end_time - start_time:.1f} s',
                flush=True,
            )

        if stop_signal.item():
            break

    if rank == 0:
        print(f'Total training time: {time.perf_counter() - start_time_train:.1f} s')
        run.finish()

    ###################### EVALUATION #########################
    # Reload the best checkpoint and reuse validate()
    model, _, _, loss_fn, _, _, _ = get_train_objs(
        cfg, n_classes, checkpoint_file, wandb_kwargs
    )
    model.to(local_rank, memory_format=torch.channels_last)
    model = DDP(model, device_ids=[local_rank])
    val_loss, val_acc1, val_acc5 = validate(model, dataloader_val, loss_fn, local_rank)
    val_loss, val_acc1, val_acc5 = utils.reduce(
        [val_loss, val_acc1, val_acc5], local_rank
    ).tolist()
    if rank == 0:
        print(f'Top 1 accuracy on validation set: {val_acc1:.2f}')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
