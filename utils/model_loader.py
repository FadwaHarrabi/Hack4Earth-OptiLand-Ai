import torch
from torchvision import models
from torchsummary import summary

def create_resnet50_model(num_classes, device):
    """
    Create and configure a ResNet-50 model for a specific number of classes.

    This function loads a pre-trained ResNet-50 model, modifies the final 
    fully connected layer to match the specified number of output classes, 
    and moves the model to the specified device (CPU or GPU). Additionally, 
    it prints a summary of the model architecture.

    Args:
        num_classes (int): The number of output classes for the classification task.
        device (str): The device to which the model will be moved. Should be 
                      either 'cpu' or 'cuda'.

    Returns:
        torch.nn.Module: The modified ResNet-50 model ready for training or inference.
    """
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify the final fully connected layer to match the number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Move the model to the specified device (CPU or GPU)
    model = model.to(device)
    
    # Print the model summary
    summary(model, (3, 224, 224))
    
    return model
