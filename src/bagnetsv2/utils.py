import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm

IMAGENET_NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

VALID_DATASETS = ('imagenet', 'imagenette')
VALID_MODEL_VARIANTS = ('bagnet9', 'bagnet17', 'bagnet33', 'resnet18', 'resnet50')
VALID_OPTIMIZERS = ('sgd', 'adamw')


def validate_config(cfg) -> None:
    """Fail fast on configuration errors."""
    if cfg.dataset.name not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset.name '{cfg.dataset.name}', expected one of {VALID_DATASETS}"
        )
    if cfg.model.variant not in VALID_MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model.variant '{cfg.model.variant}', expected one of {VALID_MODEL_VARIANTS}"
        )
    if cfg.optim.name not in VALID_OPTIMIZERS:
        raise ValueError(
            f"Unknown optim.name '{cfg.optim.name}', expected one of {VALID_OPTIMIZERS}"
        )


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_tf32(enabled: bool) -> None:
    """Enable or disable TF32 for both cuDNN convolutions and cuBLAS matmuls.

    TF32 requires Ampere or newer (compute capability >= 8.0, e.g. A100).
    On older GPUs (e.g. V100) these flags are silently ignored.
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            name = torch.cuda.get_device_name()
            warnings.warn(
                f'TF32 requires compute capability >= 8.0 (Ampere+), but the current '
                f"device '{name}' has capability {major}.x. This setting will have no "
                f'effect and all matmuls/convolutions will run in FP32.'
            )

    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled


def load_train_dataset(dir, name, transform, val_size=0.05, seed=42):
    if name == 'imagenet':
        dataset_train = datasets.ImageNet(
            dir, split='train', transform=transform['train']
        )
        dataset_val = datasets.ImageNet(dir, split='train', transform=transform['test'])
        n_classes = 1000
    else:
        dataset_train = datasets.Imagenette(
            dir,
            split='train',
            transform=transform['train'],
            size='320px',
            download=True,
        )
        dataset_val = datasets.Imagenette(
            dir,
            split='train',
            transform=transform['test'],
            size='320px',
            download=True,
        )
        n_classes = 10

    # Split training dataset into training and validation set with different transforms
    dataset_train, dataset_val = split_imagenet_train_val(
        dataset_train, dataset_val, val_size=val_size, seed=seed
    )
    return dataset_train, dataset_val, n_classes


def load_eval_dataset(dir, name, transform):
    if name == 'imagenet':
        dataset_test = datasets.ImageNet(dir, split='val', transform=transform)
        n_classes = 1000
    else:
        dataset_test = datasets.Imagenette(
            dir, split='val', transform=transform, size='320px', download=True
        )
        n_classes = 10
    return dataset_test, n_classes


def split_imagenet_train_val(dataset_train, dataset_val, val_size=0.1, seed=42):
    """Divide imagenet train into training and validation set with the same label distribution"""
    targets = (
        dataset_train.targets
        if hasattr(dataset_train, 'targets')
        else [y for _, y in dataset_train]
    )
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset_train)),
        test_size=val_size,
        stratify=targets,
        random_state=seed,
    )
    dataset_train = Subset(dataset_train, train_idx)
    dataset_val = Subset(dataset_val, val_idx)
    return dataset_train, dataset_val


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_augmentations(img_size, normalization=None, imagenet=False):
    if imagenet:
        transform = {
            # Regular augmentations for imagenet
            'train': transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            'test': transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                ]
            ),
        }
    else:
        # Augmentations for fundus images
        transform = {
            'train': transforms.Compose(
                [
                    # transforms.RandomResizedCrop(size=img_size, scale=[0.9, 1.1], ratio=[0.9, 1.1]),
                    transforms.RandomRotation(degrees=(-15, 15)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.2, 0.2, 0, 0)], p=0.5
                    ),
                    transforms.ToTensor(),
                ]
            ),
            'test': transforms.Compose([transforms.ToTensor()]),
        }

    if normalization:
        normalize = transforms.Normalize(normalization['mean'], normalization['std'])
        _ = [transform[k].transforms.append(normalize) for k in transform]

    return transform


class EarlyStopping:
    """
    Args:
        patience (int): number of epochs to wait for improvement.
        min_delta (float): minimum change to qualify as an improvement.
        checkpoint_file (str): filename to save the best checkpoint.
        verbose (bool): print a message when a checkpoint is saved and the training is stopped.
    """

    def __init__(
        self, patience=5, min_delta=0.1, checkpoint_file='model.pt', verbose=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_file = checkpoint_file
        self.verbose = verbose
        self.best_loss = np.inf
        self.no_improvement_count = 0
        self.stop_training = False

    def __call__(self, val_loss, checkpoint):
        if (self.best_loss is None) or (val_loss < self.best_loss - self.min_delta):
            self.save_checkpoint(val_loss, checkpoint)
            self.no_improvement_count = 0
            self.best_loss = val_loss
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print(
                        f'Stopping early after {self.patience} epochs with no improvement.'
                    )

    def save_checkpoint(self, val_loss, checkpoint):
        """Save checkpoint when validation loss decreases."""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.best_loss:.3f} → {val_loss:.3f}). Saving checkpoint...'
            )

        torch.save(checkpoint, self.checkpoint_file)


def get_optimizer(name, params, lr, weight_decay):
    """Build an SGD or AdamW optimizer.

    Args:
        name (str): 'sgd' or 'adamw'.
        params: iterable of parameters to optimize (e.g. model.parameters()).
        lr (float): learning rate.
        weight_decay (float): weight decay.

    Returns:
        torch.optim.Optimizer
    """
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{name}', use 'sgd' or 'adamw'")


def ddp_setup():
    """DDP setup for multiGPU training scripts.

    Returns:
        tuple[int, int, int]: global rank, local rank and world size, read from the
            environment variables set by torchrun.
    """
    local_rank = int(os.environ['LOCAL_RANK'])  # Provided by torchrun
    torch.cuda.set_device(local_rank)  # rank = unique identifier of each process

    rank = int(os.environ['RANK'])  # global rank
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f'cuda:{local_rank}'),
    )
    return rank, local_rank, world_size


def reduce(value, device):
    """Average a scalar value across all DDP processes.

    Args:
        value (float or torch.Tensor): value to reduce, computed locally on this rank.
        device: device to move value to before the collective all_reduce call.

    Returns:
        torch.Tensor: the value averaged across all processes.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value = value / dist.get_world_size()
    return value


def init_experiment(cfg, rank=None):
    """Initialize a training experiment with Hydra and (optional) wandb.

    Args:
        cfg (DictConfig): resolved Hydra config for the run.
        rank (int, optional): process rank when running under DDP, None for single-GPU
            runs. Only rank 0 (or None) creates the checkpoint directory.

    Returns:
        tuple[dict, pathlib.Path]: kwargs to pass to wandb.init(), and the checkpoint
            file path (checkpoints/<experiment_name>.pt).
    """
    hydra_cfg = HydraConfig.get()
    experiment_dir = Path(hydra_cfg.runtime.output_dir)

    wandb_kwargs = {
        'project': cfg.wandb.project,
        'entity': cfg.wandb.entity,
        'dir': experiment_dir,
        'name': cfg.experiment_name,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'mode': cfg.wandb.mode,
    }

    checkpoint_file = Path.cwd().joinpath('checkpoints', f'{cfg.experiment_name}.pt')
    if rank in (0, None):
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    return wandb_kwargs, checkpoint_file


def checkpoint2weights(checkpoint_file, model_file):
    # Load checkpoint and save only state_dict
    checkpoint = torch.load(
        checkpoint_file, map_location=torch.device('cpu'), weights_only=False
    )
    torch.save(checkpoint['state_dict'], model_file)


def predict(model, dataloader, device, autocast=True):
    """Run model inference over a dataloader.

    Args:
        model: model to evaluate.
        dataloader: dataloader to iterate over.
        device: device to run inference on.
        autocast (bool): run the forward pass under torch.autocast(fp16). Disable to
            compare to fp32/tf32.

    Returns:
        tuple[np.ndarray, np.ndarray]: softmax probabilities and targets.
    """
    model.to(device)
    model.eval()

    probs, targets = [], []
    for b, (img, labels) in enumerate(tqdm(dataloader)):
        img, labels = img.to(device), labels.to(device)

        with (
            torch.no_grad(),
            torch.autocast(device_type='cuda', dtype=torch.float16, enabled=autocast),
        ):
            y = model(img)
            p = torch.nn.functional.softmax(y, dim=1)

        probs.extend(p.tolist())
        targets.extend(labels.tolist())

    probs = np.array(probs)
    targets = np.array(targets)

    return probs, targets


def predict_logits(model, dataloader, device, autocast=True):
    """Same as predict(), but returns raw model outputs instead of softmax probabilities.

    Args:
        model: model to evaluate.
        dataloader: dataloader to iterate over.
        device: device to run inference on.
        autocast (bool): run the forward pass under torch.autocast(fp16). Disable to
            compare to fp32/tf32.

    Returns:
        tuple[np.ndarray, np.ndarray]: raw model outputs (logits) and targets.
    """
    model.to(device)
    model.eval()

    preds, targets = [], []
    for b, (img, labels) in enumerate(tqdm(dataloader, disable=tqdm_disable())):
        img, labels = img.to(device), labels.to(device)

        with (
            torch.no_grad(),
            torch.autocast(device_type='cuda', dtype=torch.float16, enabled=autocast),
        ):
            y = model(img)

        preds.extend(y.tolist())
        targets.extend(labels.tolist())

    preds = np.array(preds).squeeze()
    targets = np.array(targets)

    return preds, targets


def tqdm_disable() -> bool:
    """Whether to disable tqdm bars: true when stdout isn't a real terminal.

    In a SLURM job, stdout is redirected to a `.out` file, so each refresh
    floods the log with its own line instead of overwriting in place.
    """
    return not sys.stdout.isatty()


if __name__ == '__main__':
    # checkpoint2weights(
    #     'checkpoints/bagnet33_imagenet_pretrained.pt',
    #     'models/bagnet33_imagenet_pretrained.pt',
    # )
    print('utils.py')
