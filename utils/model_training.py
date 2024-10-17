# model_training.py

import torch
import numpy as np
from tqdm import tqdm
import os

def evaluate(model, dataloader, criterion, device, phase="val"):
    """
    Evaluate the model on the given dataloader and compute loss and accuracy.

    This function sets the model to evaluation mode, processes the inputs 
    through the model, computes the loss and accuracy, and prints the 
    results.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation or test dataset.
        criterion (torch.nn.Module): Loss function to compute the loss.
        device (str): The device to which the model and data will be moved ('cpu' or 'cuda').
        phase (str): The phase of evaluation ('val' for validation or 'test' for testing).

    Returns:
        tuple: A tuple containing the average loss and accuracy for the evaluation phase.
    """
    model.eval()

    running_loss = 0.0
    running_total_correct = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():  # Disable gradient calculation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_total_correct += torch.sum(preds == labels).item()

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = (running_total_correct / len(dataloader.dataset)) * 100
    print(f"{phase.title()} Loss: {epoch_loss:.2f}; Accuracy: {epoch_accuracy:.2f}")

    return epoch_loss, epoch_accuracy


def train(model, dataloader, criterion, optimizer, device):
    """
    Train the model on the given dataloader for one epoch.

    This function sets the model to training mode, processes the inputs, 
    computes the loss, performs backpropagation, and updates the model weights.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        criterion (torch.nn.Module): Loss function to compute the loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model weights.
        device (str): The device to which the model and data will be moved ('cpu' or 'cuda').

    Returns:
        tuple: A tuple containing the average loss and accuracy for the training epoch.
    """
    model.train()

    running_loss = 0.0
    running_total_correct = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Compute the gradients wrt the loss
        loss.backward()

        # Update the weights
        optimizer.step()

        # Calculate statistics
        _, preds = torch.max(outputs, 1)

        # Calculate running loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        running_total_correct += torch.sum(preds == labels).item()

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = (running_total_correct / len(dataloader.dataset)) * 100
    print(f"Train Loss: {epoch_loss:.2f}; Accuracy: {epoch_accuracy:.2f}")

    return epoch_loss, epoch_accuracy


def fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer, device):
    """
    Train and validate the model over a specified number of epochs.

    This function iterates over the training and validation sets for the 
    given number of epochs, keeping track of the best model based on 
    validation loss.

    Args:
        model (torch.nn.Module): The model to train and validate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        n_epochs (int): The number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        criterion (torch.nn.Module): Loss function to compute the loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model weights.
        device (str): The device to which the model and data will be moved ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The best model after training and validation.
    """
    best_loss = np.inf
    best_model = None

    # Train and validate over n_epochs
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device, phase="val")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    return best_model


def save_model(best_model, model_file):
    """
    Save the trained model to a specified file.

    This function saves the model's state dictionary to the given file path.

    Args:
        best_model (torch.nn.Module): The trained model to save.
        model_file (str): The file path where the model will be saved.

    Returns:
        None
    """
    torch.save(best_model.state_dict(), model_file)
    print(f'Model successfully saved to {model_file}.')
