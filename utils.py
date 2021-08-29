import os

import numpy as np
import pandas as pd
import torch

import config


def make_csv(path, name):
    """
    Makes csv file with images and their labels.

    Arguments:
        path (string): path to the images;
        name (string): name for csv file.
    """
    df = None

    folders = os.listdir(path)
    for folder in folders:
        images = os.listdir(os.path.join(path, folder))
        if df is None:
            df = pd.DataFrame({'image_name': images, 'folder': [folder for _ in range(len(images))]})
        else:
            to_concat = pd.DataFrame({'image_name': images, 'folder': [folder for _ in range(len(images))]})
            df = pd.concat([df, to_concat])

    map_dict = {folder: i for folder, i in zip(folders, range(len(folders)))}
    df['label'] = df['folder'].map(map_dict)

    df.to_csv(name, index=False)


def compute_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader
    
    Returns: 
        accuracy (float): a float value between 0 and 1.
    """
    model.eval()
    
    correct = 0
    total = 0
    for x, y, _ in loader:
        x_gpu = x.to(config.DEVICE)
        y_gpu = y.to(config.DEVICE)

        preds = torch.argmax(model(x_gpu), 1)
        correct += torch.sum(preds == y_gpu)
        total += y.shape[0]

    accuracy = correct / total
    
    return accuracy


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    model.load_state_dict(checkpoint["state_dict"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == '__main__':
    make_csv('./train', 'train.csv')
    make_csv('./val', 'val.csv')
