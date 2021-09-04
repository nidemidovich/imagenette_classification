import PIL
import torch
from torchvision import transforms as T


BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
FACTOR = 0.333
PATIENCE = 3
NUM_EPOCHS = 15
CHECKPOINT_FILE = "resnet18.pth.tar"
SAVE_MODEL = True
LOAD_MODEL = True

# for baseline and test data, without augmentation
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# augmentations
augmentations = T.Compose([
    T.Resize((224, 224)),
    T.RandomCrop((210, 210)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.ColorJitter(hue=0.2, saturation=0.2, brightness=0.3),
    T.RandomRotation(10, resample=PIL.Image.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])