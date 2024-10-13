if __name__ == '__main__':

    from utils.data_preprocessing import create_transforms, split_dataset
    from utils.visualization import visualize_data
    from utils.model_loader import create_resnet50_model
    from utils.model_training import evaluate, train, fit, save_model
    import torch
    import os 

    # Parameters
    data_directory = './data/raw/'  # Path to the raw dataset
    input_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    batch_size = 16
    num_workers = 2
    train_size = 0.70
    val_size = 0.15

    # Step 1: Create transforms
    train_transform, val_transform, test_transform = create_transforms(
        input_size, imagenet_mean, imagenet_std
    )

    # Step 2: Load and split dataset into train/validation/test
    train_loader, val_loader, test_loader, class_names = split_dataset(
        data_directory, train_transform, val_transform, test_transform, 
        train_size=train_size, val_size=val_size, batch_size=batch_size, num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visualize_data(train_loader, class_names, imagenet_std, imagenet_mean, n=3)

    # Hyperparameters
    n_epochs = 20
    lr = 1e-3
    model = create_resnet50_model()

    # Specify criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_model = fit(model, train_loader, val_loader, n_epochs, lr, criterion, optimizer, device)

    # Evaluate on the test set
    test_loss, test_accuracy = evaluate(best_model, test_loader, criterion, device, phase="test")

    # Save the model
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file = os.path.join(model_dir, 'best_model.pth')
    save_model(best_model, model_file)
