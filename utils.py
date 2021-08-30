import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

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
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        preds = torch.argmax(model(x), 1)
        correct += torch.sum(preds == y)
        total += y.shape[0]

    accuracy = correct / total
    
    return accuracy


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr):
    model.load_state_dict(checkpoint["state_dict"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class SubsetSampler(Sampler):
    """
    Samples elements with given indices sequentially

    Arguments:
        data_source (Dataset): dataset to sample from
        indices (ndarray): indices of the samples to take
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    
def evaluate_model(model, dataset, indices):
    """
    Computes predictions and ground truth labels for the indices of the dataset
    
    Returns: 
        predictions: np array of booleans of model predictions
        grount_truth: np array of boolean of actual labels of the dataset
    """
    model.eval()
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE, 
        sampler=SubsetSampler(indices)
    )
    
    predictions = []
    ground_truth = []
    for x, y, _ in loader:
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        predictions += torch.argmax(model(x), 1).tolist()
        ground_truth += y.tolist()
    
    return np.array(predictions), np.array(ground_truth)


def visualize_predictions(model, train_dataset, orig_dataset, classes, count=10):
    """
    Visualize random number of predictions (10 by default).
    """
    indices = np.random.choice(range(len(train_dataset)), size=10, replace=False)

    plt.figure(figsize=(count * 3, 3))
    for i, idx in enumerate(indices):
        x, _, _ = train_dataset[idx]
        x = x.to(config.DEVICE)

        preds = torch.argmax(model(torch.unsqueeze(x, 0)), 1)

        plt.subplot(1, 10, i + 1)
        plt.title(f'Label: {classes[float(preds)]}')
        plt.imshow(orig_dataset[idx][0])
        plt.grid(False)
        plt.axis('off')


def visualize_samples(dataset, title=None, count=10):
    """
    Visualize random number of samples (10 by default).
    """
    indices = np.random.choice(np.arange(len(dataset)), count, replace=False)

    plt.figure(figsize=(count*3, 3))
    if title:
        plt.suptitle(title)        
    for i, index in enumerate(indices):    
        x, y, _ = dataset[index]
        plt.subplot(1, count, i + 1)
        plt.title("Label: %s" % y)
        plt.imshow(x)
        plt.grid(False)
        plt.axis('off') 
