import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(batch_size):
    """
    Prepares the SVHN data loaders for training and testing with data augmentation.
    """
    # # SEPARATE TRANSFORMS: Augmentation for train, simple for test
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),           # Random crop with padding
    #     transforms.RandomHorizontalFlip(p=0.5),         # 50% chance of horizontal flip
    #     transforms.ColorJitter(                         # Random color changes
    #         brightness=0.2, 
    #         contrast=0.2, 
    #         saturation=0.1,
    #         hue=0.05
    #     ),
    #     transforms.RandomRotation(5),                   # Small random rotations
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomRotation(10),  # Increased from 5
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test transform - no augmentation, just normalize
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Apply different transforms to train and test sets
    trainset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4  # Increased workers
    )
    
    testset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    print(f"Training samples: {len(trainset)}, Test samples: {len(testset)}")
    return trainloader, testloader

def plot_training_history(output_folder, history):
    """
    Plots the training and validation loss and accuracy with improved styling.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    ax1.plot(history['test_loss'], label='Test Loss', color='orange', alpha=0.7)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue', alpha=0.7)
    ax2.plot(history['test_acc'], label='Test Accuracy', color='orange', alpha=0.7)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {os.path.join(output_folder, 'training_history.png')}")

def plot_predictions(model, data_loader, device, output_folder, num_images=10):
    """
    Displays a grid of test images with their predicted and true labels.
    """
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    fig = plt.figure(figsize=(15, 7))
    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        
        # Un-normalize the image
        img = images[i] / 2 + 0.5
        img = np.clip(img, 0, 1)  # Ensure values are in [0,1]
        img = np.transpose(img, (1, 2, 0))
        
        ax.imshow(img)
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {predicted[i]}\nTrue: {labels[i]}", color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'test_predictions.png'), dpi=300, bbox_inches='tight')
    print(f"Saved sample predictions to {os.path.join(output_folder, 'test_predictions.png')}")

def visualize_augmentations(dataset, output_folder, num_samples=8):
    """
    Visualize the effect of data augmentation on sample images.
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 4))
    
    for i in range(num_samples):
        img, label = dataset[i]
        
        # Convert tensor to numpy for visualization
        img_np = img / 2 + 0.5  # Un-normalize
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))
        
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Label: {label}")
        axes[0, i].axis('off')
        
        # Get another augmented version
        img2, _ = dataset[i]
        img2_np = img2 / 2 + 0.5
        img2_np = np.clip(img2_np, 0, 1)
        img2_np = np.transpose(img2_np, (1, 2, 0))
        
        axes[1, i].imshow(img2_np)
        axes[1, i].set_title("Augmented")
        axes[1, i].axis('off')
    
    plt.suptitle("Original vs Augmented Images")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'augmentation_examples.png'), dpi=300, bbox_inches='tight')
    print(f"Saved augmentation examples to {os.path.join(output_folder, 'augmentation_examples.png')}")