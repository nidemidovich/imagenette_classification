import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torchvision import models

from dataset import ImagenetteDataset
import config
from utils import compute_accuracy


def train_model(model, train_loader, val_loader, loss, optimizer, num_epochs, scheduler=None, scheduler_loss=False): 
    """
    Trains model and accumulate loss history, train history and val history.
    """   
    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train()
        
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        for i_step, (x, y, _) in enumerate(train_loader):
          
            x_gpu = x.to(config.DEVICE)
            y_gpu = y.to(config.DEVICE)
            prediction = model(x_gpu)    
            loss_value = loss(prediction, y_gpu)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y_gpu)
            total_samples += y.shape[0]
            
            loss_accum += loss_value

        ave_loss = loss_accum / i_step
        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(model, val_loader)
        
        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)

        if scheduler:
            if scheduler_loss:
                scheduler.step(ave_loss)
            else:
                scheduler.step()
        
        print(f'Average loss: {ave_loss}, Train accuracy: {train_accuracy}, Val accuracy: {val_accuracy}')
        
    return loss_history, train_history, val_history


def main():
    train_dataset = ImagenetteDataset(
        csv_file='train.csv',
        root_dir='train',
        transforms=config.transforms
    )
    val_dataset = ImagenetteDataset(
        csv_file='val.csv',
        root_dir='val',
        transforms=config.transforms
    )

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)

    train_indices = list(range(train_data_size))
    val_indices = list(range(val_data_size))

    np.random.seed(42)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=val_sampler
    )

    model = models.resnet18(pretrained=True)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model = model.to(config.DEVICE)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(), 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        factor=config.FACTOR, 
        patience=config.PATIENCE
    )

    loss_history, train_history, val_history = train_model(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        config.NUM_EPOCHS,
        scheduler,
        True
    )
    