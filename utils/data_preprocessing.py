import os
import shutil
import torch
import numpy as np
from torch.utils import data
from torchvision import datasets, transforms

# Custom Dataset class for EuroSAT
class EuroSAT(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading the EuroSAT dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to wrap.
        transform (callable, optional): A function/transform to apply to the data.

    Methods:
        __getitem__(index): Returns a single sample from the dataset.
        __len__(): Returns the total number of samples in the dataset.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)

# Function to create transformations
def create_transforms(input_size=224, mean=None, std=None):
    """
    Create data transformations for training, validation, and testing.

    Args:
        input_size (int): The target size for resizing the images.
        mean (list, optional): The mean values for normalization. 
                               Defaults to ImageNet statistics if not provided.
        std (list, optional): The standard deviation values for normalization.
                              Defaults to ImageNet statistics if not provided.

    Returns:
        tuple: A tuple containing the training, validation, and testing transformations.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, val_transform, test_transform

# Function to load and split the dataset
def split_dataset(data_dir, train_transform, val_transform, test_transform, 
                  train_size=0.70, val_size=0.15, batch_size=16, num_workers=2, processed_dir='data/processed'):
    """
    Load and split the dataset into training, validation, and testing sets.

    Args:
        data_dir (str): The directory containing the dataset.
        train_transform (callable): Transformations to apply to the training set.
        val_transform (callable): Transformations to apply to the validation set.
        test_transform (callable): Transformations to apply to the test set.
        train_size (float): Proportion of the dataset to use for training. 
                            Defaults to 0.70.
        val_size (float): Proportion of the dataset to use for validation. 
                          Defaults to 0.15.
        batch_size (int): Number of samples per batch. Defaults to 16.
        num_workers (int): Number of subprocesses to use for data loading. 
                           Defaults to 2.
        processed_dir (str): Directory to save processed data. Defaults to 'data/processed'.

    Returns:
        tuple: A tuple containing DataLoaders for the training, validation, 
               and testing sets, along with the class labels.
    """
    # Load the dataset without transforms (transform will be applied to the subsets)
    dataset = datasets.ImageFolder(data_dir)
    
    # Randomly shuffle the dataset and split into train, validation, and test
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    train_split = int(train_size * len(dataset))
    val_split = int(val_size * len(dataset))
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:train_split + val_split]
    test_indices = indices[train_split + val_split:]
    
    # Subset the dataset with appropriate transforms
    train_data = data.Subset(EuroSAT(dataset, train_transform), train_indices)
    val_data = data.Subset(EuroSAT(dataset, val_transform), val_indices)
    test_data = data.Subset(EuroSAT(dataset, test_transform), test_indices)
    
    # Create DataLoader for each subset
    train_loader = data.DataLoader(train_data, batch_size=batch_size, 
                                   num_workers=num_workers, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size=batch_size, 
                                 num_workers=num_workers, shuffle=False)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, 
                                  num_workers=num_workers, shuffle=False)
    
    # Print dataset sizes
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    # Save the data splits into 'processed' folder
    #save_split_data(dataset, train_indices, val_indices, test_indices, processed_dir)

    return train_loader, val_loader, test_loader, dataset.classes

# Function to save the split data into train/val/test folders
def save_split_data(dataset, train_indices, val_indices, test_indices, processed_dir):
    """
    Save the split dataset into separate train, validation, and test folders.

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing the samples.
        train_indices (list): Indices of training samples.
        val_indices (list): Indices of validation samples.
        test_indices (list): Indices of test samples.
        processed_dir (str): Directory to save the processed data.
    """
    # Create directories
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Helper function to copy images to their respective folders
    def copy_images(indices, split_dir):
        for idx in indices:
            img_path, label = dataset.samples[idx]
            label_dir = os.path.join(split_dir, dataset.classes[label])
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(img_path, label_dir)

    # Copy images to train/val/test folders
    print(f"Saving train data to {train_dir}...")
    copy_images(train_indices, train_dir)

    print(f"Saving validation data to {val_dir}...")
    copy_images(val_indices, val_dir)

    print(f"Saving test data to {test_dir}...")
    copy_images(test_indices, test_dir)

    print("Data successfully saved into 'processed' directory")
