import matplotlib.pyplot as plt
import numpy as np

def visualize_data(train_loader, class_names, imagenet_std, imagenet_mean, n=3):
    # Get a batch of images and their corresponding labels
    inputs, classes = next(iter(train_loader))
    
    # Set up the plot
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    
    # Denormalize the images and plot them
    for i in range(n):
        for j in range(n):
            idx = i * n + j  # Flatten the grid index to access the batch elements
            if idx >= len(inputs):
                break  # Avoid accessing out-of-range elements
            
            # Convert tensor to numpy and denormalize
            image = inputs[idx].numpy().transpose((1, 2, 0))
            image = np.clip(np.array(imagenet_std) * image + np.array(imagenet_mean), 0, 1)

            # Get the class label
            title = class_names[classes[idx]]
            
            # Plot the image
            axes[i, j].imshow(image)
            axes[i, j].set_title(title)
            axes[i, j].axis('off')  # Turn off axis labels and ticks
    
    plt.tight_layout()  # Adjust layout to fit titles
    plt.show()
