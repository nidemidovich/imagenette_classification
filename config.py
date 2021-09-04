import torch
from torchvision import transforms as t


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

transforms = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
