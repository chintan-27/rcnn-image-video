import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(batch_size):
    """
    Prepares the SVHN data loaders for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # The SVHN dataset labels '10' as '0'. We need to map it to the correct index.
    # The dataloader handles this automatically by using labels 0-9.

    return trainloader, testloader

def plot_training_history(output_folder, history):
    """
    Plots the training and validation loss and accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epochs')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_history.png'))
    print("\nSaved training history plot to training_history.png")

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
    
    fig = plt.figure(figsize=(15, 7))
    for i in range(num_images):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        # Un-normalize the image
        img = images[i] / 2 + 0.5 
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)
        
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {predicted[i].item()}\nTrue: {labels[i].item()}", color=color)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'test_predictions.png'))
    print("Saved sample predictions to test_predictions.png")


