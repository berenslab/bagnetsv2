import time

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import get_model as tv_get_model

# Use bagnetsv2.bagnets to see the original model's results
from bagnetsv2 import bagnetsv2 as bagnets
from bagnetsv2 import utils


def get_dataloader(cfg):
    image_size = (cfg.train.image_size, cfg.train.image_size)
    transform = utils.get_augmentations(
        image_size, normalization=utils.IMAGENET_NORMALIZATION, imagenet=True
    )['test']

    dataset_test, n_classes = utils.load_eval_dataset(
        cfg.dataset.dir, cfg.dataset.name, transform
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader_test, n_classes


def load_model(cfg, n_classes):
    if 'bagnet' in cfg.model.variant:
        model = bagnets.get_bagnet(
            cfg.model.variant, weights=None, num_classes=n_classes
        )
    else:
        model = tv_get_model(cfg.model.variant, weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    checkpoint = torch.load(cfg.checkpoint, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model


@hydra.main(version_base=None, config_path='../../configs', config_name='default')
def main(cfg: DictConfig):
    start = time.perf_counter()
    utils.validate_config(cfg)

    dataloader_test, n_classes = get_dataloader(cfg)
    model = load_model(cfg, n_classes)

    # Check for dead convolutional layers
    dead_layer_count = 0
    for name, parameters in model.named_parameters():
        if 'conv' in name:
            max_weight = parameters.flatten().abs().max()

            if max_weight <= 1e-4:
                dead_layer_count += 1

    print(f'Dead layer count (max(abs(parameters) <= 1e-4 ) = {dead_layer_count}')

    # Accuracy
    probs, targets = utils.predict(model, dataloader_test, cfg.device)

    acc = utils.accuracy(torch.from_numpy(probs), torch.from_numpy(targets), (1, 5))
    print(f'Top 1 accuracy on validation set: {acc[0].item():.2f}')
    print(f'Top 5 accuracy on validation set: {acc[1].item():.2f}')

    print(f'Total testing time: {time.perf_counter() - start:.1f} s')


if __name__ == '__main__':
    main()
