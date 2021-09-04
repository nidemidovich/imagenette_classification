import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import models

from dataset import ImagenetteDataset
import config
from utils import compute_accuracy, load_checkpoint, save_checkpoint


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
          
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)
            prediction = model(x)    
            loss_value = loss(prediction, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            _, indices = torch.max(prediction, 1)
            correct_samples += torch.sum(indices == y)
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

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"resnet18_{epoch}.pth.tar")
        
        print(f'Average loss: {ave_loss}, Train accuracy: {train_accuracy}, Val accuracy: {val_accuracy}')
        
    return loss_history, train_history, val_history


def conduct_training():
    train_dataset = ImagenetteDataset(
        csv_file='train.csv',
        root_dir='train',
        transforms=config.transforms
    )
    test_dataset = ImagenetteDataset(
        csv_file='val.csv',
        root_dir='val',
        transforms=config.transforms,
        train=False
    )

    train_data_size = len(train_dataset)

    val_split = int(np.floor(0.2 * train_data_size))

    indices = list(range(train_data_size))

    np.random.seed(42)
    np.random.shuffle(indices)

    val_indices, train_indices = indices[:val_split], indices[val_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=train_dataset, 
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
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        factor=config.FACTOR, 
        patience=config.PATIENCE
    )
    
    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

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
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config.BATCH_SIZE
    )
    test_accuracy = compute_accuracy(model, test_loader)

    return loss_history, train_history, val_history, test_accuracy