import os

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import config


class ImagenetteDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms=None, train=True):
        """
        Arguments:
            csv_file (string): csv file with each image and its label;
            root_dir (string): directory with all the images;
            transforms (callable, optional): optional transform to be applied on a sample.
        """
        super().__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        record = self.annotations.iloc[index]

        image_name = record['image_name']
        folder = record['folder']
        label = record['label']

        img_file = os.path.join(self.root_dir, folder, image_name)
        img = Image.open(img_file)
        img = img.convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        if not self.train:
            return img

        return img, label, image_name


if __name__ == '__main__':
    dataset = ImagenetteDataset(
        csv_file='train.csv',
        root_dir='train',
        transforms=config.transforms
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=32
    )

    for x, y, file in loader:
        print(x.shape)
        print(y.shape)
        break
