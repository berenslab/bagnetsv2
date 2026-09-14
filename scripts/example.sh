#!/bin/sh

set -e

echo Training bagnet33 on Imagenette (single GPU)...
python -m bagnetsv2.train model.variant=bagnet33 dataset.name=imagenette train.epochs=90

echo Training bagnet33 on ImageNet (multi-GPU)...
torchrun --standalone --nproc_per_node=8 -m bagnetsv2.train_multigpu model.variant=bagnet33 dataset.name=imagenet dataset.dir=<IMAGENET_DIR> train.batch_size=1024 optim.lr=0.04 train.epochs=90 train.num_workers=4

echo Evaluating the trained checkpoint...
python -m bagnetsv2.eval model.variant=bagnet33 dataset.name=imagenet dataset.dir=/path/to/ImageNet2012
