# 필요한 라이브러리
import time
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
import os

##### Function for training and evaluating a model
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device):
    """
    Args:
        model: Model to be trained
        criterion: Loss function
        optimizer: Optimizer to use for training
        scheduler: Learning rate scheduler
        train_loader: DataLoader for training dataset
        val_loader: DataLoader for validation dataset
        num_epochs: Number of training epochs
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Trained model with the best weights
        epoch_lst: List of epoch indices
        trn_metadata: Training loss and accuracy over epochs
        val_metadata: Validation loss and accuracy over epochs
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    epoch_lst = []
    trn_loss_lst = []
    trn_acc_lst = []
    val_loss_lst = []
    val_acc_lst = []

    for epoch in range(num_epochs):
        print('-' * 50)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluation mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Save training and validation metrics
            if phase == 'train':
                epoch_lst.append(epoch)
                trn_loss_lst.append(np.round(epoch_loss, 4))
                trn_acc_lst.append(np.round(epoch_acc.cpu().item(), 4))
            elif phase == 'val':
                val_loss_lst.append(np.round(epoch_loss, 4))
                val_acc_lst.append(np.round(epoch_acc.cpu().item(), 4))

            # Deep copy the model if it's the best on validation
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'model_best.pt')  # Save best model weights

        # Save latest model weights after every epoch
        torch.save(model.state_dict(), 'model_latest.pt')

    # Training metadata
    trn_metadata = [trn_loss_lst, trn_acc_lst]
    val_metadata = [val_loss_lst, val_acc_lst]

    # Training complete
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_lst, trn_metadata, val_metadata, best_epoch
