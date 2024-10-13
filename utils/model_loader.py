import torch
from torchvision import models
from torchsummary import summary

def create_resnet50_model(num_classes, device):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Modify the final fully connected layer to match the number of classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Move the model to the specified device (CPU or GPU)
    model = model.to(device)
    
    # Print the model summary
    summary(model, (3, 224, 224))
    
    return model