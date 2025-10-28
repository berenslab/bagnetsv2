import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms


SEED = 42
IMAGENET_NORMALIZATION = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
IMAGENET_DIR = 'datasets/ImageNet2012'


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def split_imagenet_train_val(dataset_train, dataset_val, val_size=0.1):
    '''Divide imagenet train into training and validation set with the same label distribution'''
    targets = dataset_train.targets if hasattr(dataset_train, 'targets') else [y for _, y in dataset_train]
    train_idx, val_idx = train_test_split(np.arange(len(dataset_train)), test_size=val_size, stratify=targets, random_state=SEED)
    dataset_train = Subset(dataset_train, train_idx)
    dataset_val = Subset(dataset_val, val_idx)
    return dataset_train, dataset_val


def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
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
            'train': transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), ]), 
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(), ]) 
            }
    else:
        # Augmentations for fundus images
        transform = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(size=img_size, scale=[0.9, 1.1], ratio=[0.9, 1.1]),
                transforms.RandomRotation(degrees=(-15, 15)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0, 0)], p=0.5),
                transforms.ToTensor() ]), 
            'test': transforms.Compose([transforms.ToTensor()]) 
            }

    if normalization:
       normalize = transforms.Normalize(normalization['mean'], normalization['std'])
       _ = [transform[k].transforms.append(normalize) for k in transform.keys()]
        
    return transform


class EarlyStopping():
    '''
    Args:
        patience (int): number of epochs to wait for improvement.
        min_delta (float): minimum change to qualify as an improvement.
        checkpoint_file (str): filename to save the best checkpoint.
        verbose (bool): print a message when a checkpoint is saved and the training is stopped.
    '''
    def __init__(self, patience=5, min_delta=0.1, checkpoint_file='model.pt', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_file = checkpoint_file
        self.verbose = verbose
        self.best_loss= np.inf
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
                    print(f'Stopping early after {self.patience} epochs with no improvement.')

    def save_checkpoint(self, val_loss, checkpoint):
        '''Save checkpoint when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.3f} → {val_loss:.3f}). Saving checkpoint...')

        torch.save(checkpoint, self.checkpoint_file)


def checkpoint2weights(checkpoint_file, model_file):
    # Load checkpoint and save only state_dict
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    torch.save(checkpoint['state_dict'], model_file)


def predict(model, dataloader, device):
    model.to(device)
    model.eval()

    preds, targets = [], []
    for b, (img, labels) in enumerate(tqdm(dataloader)):            
        img, labels = img.to(device), labels.to(device)

        with torch.no_grad():
            y = model(img)
        
        preds.extend(y.tolist())
        targets.extend(labels.tolist())

    preds = np.array(preds).squeeze()
    targets = np.array(targets)

    return preds, targets

    
if __name__ == '__main__':
    print('utils.py')
    checkpoint2weights('checkpoints/bagnet33_imagenet_pretrained.pt', 'models/bagnet33_imagenet_pretrained.pt')
