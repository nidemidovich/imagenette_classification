import os

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


if __name__ == '__main__':
    make_csv('./train')
