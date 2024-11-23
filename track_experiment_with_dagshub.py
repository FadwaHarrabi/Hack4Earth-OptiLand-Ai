import os
import mlflow
import dagshub
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/fadwaharrabi58/OptiLand-Ai.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "fadwaharrabi58"  # Your DagsHub username
os.environ["MLFLOW_TRACKING_PASSWORD"] = "fb044274738a448f643b06a70d237171fa85977b"  # Your DagsHub access token

# Initialize connection to DagsHub repository
dagshub.init(
    repo_owner="fadwaharrabi58",
    repo_name="OptiLand-Ai",
    mlflow=True  # Enables MLflow tracking
)

# Paths to data and model
raw_data_path = "data/raw"  # Path containing raw data
processed_data_path = "data/processed"  # Processed data (train, test, valid)
model_path = "models/best_model.pth"  # Trained deep learning model

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for input to a deep learning model
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load datasets
train_dir = os.path.join(processed_data_path, "train")
test_dir = os.path.join(processed_data_path, "test")
valid_dir = os.path.join(processed_data_path, "val")

train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)
valid_dataset = ImageFolder(valid_dir, transform=transform)

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Define the model (ResNet-50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Use pretrained ResNet-50
model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adjust the output layer for the number of classes
model = model.to(device)


# Training setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Start an MLflow experiment
mlflow.set_experiment("OptiLand-Ai Experiment")

with mlflow.start_run():  # Begin an MLflow run
    # Log dataset paths
    mlflow.log_artifact(raw_data_path, artifact_path="raw_data")
    mlflow.log_artifact(processed_data_path, artifact_path="processed_data")

    # Log parameters
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("model_architecture", "resnet50")

    # Training loop (simplified)
    for epoch in range(1):  # Adjust the number of epochs as needed
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        mlflow.log_metric("train_loss", total_loss)
        mlflow.log_metric("train_accuracy", train_accuracy)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path)

# End of script
print("Experiment tracking complete. Check your DagsHub repository for results.")
