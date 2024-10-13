# model_training.py

import torch
import numpy as np
from tqdm import tqdm
import os

def evaluate(model, dataloader, criterion, device, phase="val"):
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
    torch.save(best_model.state_dict(), model_file)
    print(f'Model successfully saved to {model_file}.')


