import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_csv(path):
    """
    Makes csv file with images and their labels.

    Arguments:
        path (string): path to the images;
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

    df.to_csv('imagenette_dataset.csv', index=False)


def visualize_samples(dataset, title=None, count=10):
    """
    Visualizes random samples.

    Arguments:
        dataset (torch.Dataset): the dataset to visualize the samples from;
        title, (string, optional): if passed, it is set as a title of plot;
        count, (int, optional): num of the samples to visualize.
    """
    indices = np.random.choice(
        np.arange(len(dataset)),
        count,
        replace=False
    )

    plt.figure(figsize=(count * 3, 3))
    if title:
        plt.suptitle(title)
    for i, index in enumerate(indices):
        x, y, _ = dataset[index]
        plt.subplot(1, count, i + 1)
        plt.title("Label: %s" % y)
        plt.imshow(x)
        plt.grid(False)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    make_csv('./train')

