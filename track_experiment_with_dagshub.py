# import os
# import mlflow
# import dagshub
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder


# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/fadwaharrabi58/OptiLand-Ai.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "fadwaharrabi58"  # Your DagsHub username
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "fb044274738a448f643b06a70d237171fa85977b"  # Your DagsHub access token

# # Initialize connection to DagsHub repository
# dagshub.init(
#     repo_owner="fadwaharrabi58",
#     repo_name="OptiLand-Ai",
#     mlflow=True  # Enables MLflow tracking
# )

# # Paths to data and model
# raw_data_path = "data/raw"  # Path containing raw data
# processed_data_path = "data/processed"  # Processed data (train, test, valid)
# model_path = "models/best_model.pth"  # Trained deep learning model

# # Data preprocessing and augmentation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images for input to a deep learning model
#     transforms.ToTensor(),         # Convert images to PyTorch tensors
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
# ])

# # Load datasets
# train_dir = os.path.join(processed_data_path, "train")
# test_dir = os.path.join(processed_data_path, "test")
# valid_dir = os.path.join(processed_data_path, "val")

# train_dataset = ImageFolder(train_dir, transform=transform)
# test_dataset = ImageFolder(test_dir, transform=transform)
# valid_dataset = ImageFolder(valid_dir, transform=transform)

# # DataLoader setup
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# # Define the model (ResNet-50)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Use pretrained ResNet-50
# model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adjust the output layer for the number of classes
# model = model.to(device)


# # Training setup
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Start an MLflow experiment
# mlflow.set_experiment("OptiLand-Ai Experiment")

# with mlflow.start_run():  # Begin an MLflow run
#     # Log dataset paths
#     mlflow.log_artifact(raw_data_path, artifact_path="raw_data")
#     mlflow.log_artifact(processed_data_path, artifact_path="processed_data")

#     # Log parameters
#     mlflow.log_param("batch_size", 32)
#     mlflow.log_param("learning_rate", 0.001)
#     mlflow.log_param("optimizer", "adam")
#     mlflow.log_param("model_architecture", "resnet50")

#     # Training loop (simplified)
#     for epoch in range(1):  # Adjust the number of epochs as needed
#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         train_accuracy = correct / total
#         mlflow.log_metric("train_loss", total_loss)
#         mlflow.log_metric("train_accuracy", train_accuracy)

#     # Save the trained model
#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), model_path)

#     # Log the model artifact
#     mlflow.log_artifact(model_path)

# # End of script
# print("Experiment tracking complete. Check your DagsHub repository for results.")
import os
import mlflow
import dagshub
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# Environment variables for MLflow and DagsHub
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/fadwaharrabi58/OptiLand-Ai.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "fadwaharrabi58"  # Your DagsHub username
os.environ["MLFLOW_TRACKING_PASSWORD"] = "fb044274738a448f643b06a70d237171fa85977b"  # Your DagsHub access token

# Initialize DagsHub connection
dagshub.init(
    repo_owner="fadwaharrabi58",
    repo_name="OptiLand-Ai",
    mlflow=True
)

# Paths to data and model
raw_data_path = "data/raw"
processed_data_path = "data/processed"
model_path = "models/best_model.pth"

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dir = os.path.join(processed_data_path, "train")
test_dir = os.path.join(processed_data_path, "test")
valid_dir = os.path.join(processed_data_path, "val")

train_dataset = ImageFolder(train_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)
valid_dataset = ImageFolder(valid_dir, transform=transform)

# Reduce dataset size for faster experimentation
train_subset, _ = random_split(train_dataset, [1000, len(train_dataset) - 1000])
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Define the device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model (ResNet-50)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

# Training setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# For mixed precision, use GradScaler only if CUDA is available
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# Start an MLflow experiment
mlflow.set_experiment("OptiLand-Ai Experiment")

with mlflow.start_run():
    # Log dataset metadata
    mlflow.log_param("dataset_size", len(train_dataset))
    mlflow.log_param("num_classes", len(train_dataset.classes))

    # Log training parameters
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("model_architecture", "resnet50")

    # Log a small data sample
    subset_dir = "data/subset"
    os.makedirs(subset_dir, exist_ok=True)
    for i, (image, label) in enumerate(train_dataset):
        if i >= 100:  # Limit to 100 samples
            break
        image_path = os.path.join(subset_dir, f"sample_{i}.png")
        transforms.ToPILImage()(image).save(image_path)
    mlflow.log_artifact(subset_dir, artifact_path="data_sample")

    # Training loop (simplified)
    for epoch in range(15):  # Adjust the number of epochs as needed
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler is not None:  # Use mixed precision if CUDA is available
                with torch.cuda.amp.autocast():  # Enable autocasting
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()  # Scaled backward pass
                scaler.step(optimizer)  # Step optimizer with scaler
                scaler.update()  # Update the scaler
            else:  # Use standard precision if CUDA is not available
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        print(f"Epoch {epoch + 1}, Loss: {total_loss}, Accuracy: {train_accuracy:.2f}")
        mlflow.log_metric("train_loss", total_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

    # Save the trained model (compressed)
    os.makedirs("models", exist_ok=True)
    scripted_model = torch.jit.script(model)  # Convert model to TorchScript
    torch.jit.save(scripted_model, model_path)
    mlflow.log_artifact(model_path)

print("Experiment tracking complete. Check your DagsHub repository for results.")

# import os
# import random
# import mlflow
# import dagshub
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import ImageFolder
# from time import time
# from PIL import Image

# # Environment variables for MLflow and DagsHub
# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/fadwaharrabi58/OptiLand-Ai.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "fadwaharrabi58"  # Your DagsHub username
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "fb044274738a448f643b06a70d237171fa85977b"  # Your DagsHub access token

# # Initialize DagsHub connection
# dagshub.init(
#     repo_owner="fadwaharrabi58",
#     repo_name="OptiLand-Ai",
#     mlflow=True
# )

# # Paths to data and model
# processed_data_path = "data/processed"
# versioned_data_path = "data/versions"
# model_path = "models/best_model.pth"

# # Data preprocessing
# original_dataset = ImageFolder(os.path.join(processed_data_path, "train"))

# # Dataset versioning
# num_versions = 5
# images_per_version = 500
# os.makedirs(versioned_data_path, exist_ok=True)

# # Shuffle dataset for random sampling
# all_indices = list(range(len(original_dataset)))
# random.shuffle(all_indices)

# # Split and save different versions
# for version in range(1, num_versions + 1):
#     version_dir = os.path.join(versioned_data_path, f"v{version}")
#     os.makedirs(version_dir, exist_ok=True)
#     start_idx = (version - 1) * images_per_version
#     end_idx = start_idx + images_per_version
#     indices = all_indices[start_idx:end_idx]
    
#     for idx in indices:
#         image, label = original_dataset[idx]
#         label_name = original_dataset.classes[label]
#         label_dir = os.path.join(version_dir, label_name)
#         os.makedirs(label_dir, exist_ok=True)
#         image_path = os.path.join(label_dir, f"image_{idx}.png")
#         if not os.path.exists(image_path):
#             image.save(image_path)

# print("Dataset versions created successfully.")

# # MLflow experiment setup
# mlflow.set_experiment("OptiLand-Ai Experiment")

# for version in range(1, num_versions + 1):
#     version_dir = os.path.join(versioned_data_path, f"v{version}")
#     with mlflow.start_run():
#         mlflow.log_param("dataset_version", f"v{version}")
#         mlflow.log_param("num_images", images_per_version)
#         mlflow.log_artifact(version_dir, artifact_path=f"dataset/v{version}")
#         print(f"Version v{version} tracked in MLflow.")

# # Training setup
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# model.fc = torch.nn.Linear(model.fc.in_features, len(original_dataset.classes))
# model = model.to(device)

# # Training each dataset version
# epochs = 10
# batch_size = 32
# learning_rate = 0.001

# for version in range(1, num_versions + 1):
#     version_dir = os.path.join(versioned_data_path, f"v{version}")
#     train_dataset = ImageFolder(version_dir, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = torch.nn.CrossEntropyLoss()

#     with mlflow.start_run():
#         mlflow.log_param("dataset_version", f"v{version}")
#         mlflow.log_param("batch_size", batch_size)
#         mlflow.log_param("learning_rate", learning_rate)
#         mlflow.log_param("epochs", epochs)

#         for epoch in range(epochs):
#             model.train()
#             total_loss = 0
#             correct = 0
#             total = 0
#             epoch_start_time = time()

#             for images, labels in train_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 total_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#             epoch_time = time() - epoch_start_time
#             train_accuracy = correct / total

#             # Log metrics
#             mlflow.log_metric("train_loss", total_loss, step=epoch)
#             mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
#             mlflow.log_metric("epoch_time", epoch_time, step=epoch)

#             print(f"Version {version}, Epoch {epoch + 1}, Loss: {total_loss:.2f}, Accuracy: {train_accuracy:.2f}")

#         # Save and log the model
#         model_version_path = f"models/model_v{version}.pth"
#         torch.save(model.state_dict(), model_version_path)
#         mlflow.log_artifact(model_version_path)

# print("All dataset versions trained and logged.")
