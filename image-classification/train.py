import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm

from model import RCNN
from utils import get_data_loaders, plot_training_history

def train(args):
    """
    Main training function.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")

    # Make the Output Folder
    os.makedirs(args.output_folder, exist_ok=True)
    OUTPUT_FOLDER = args.output_folder

    # Load data
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Initialize model, loss, and optimizer
    model = RCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print("Starting training...")
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=running_loss/len(train_loader))

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- Validation Phase ---
        model.eval()
        running_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            progress_bar_test = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Test]")
            for inputs, labels in progress_bar_test:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                progress_bar_test.set_postfix(loss=running_loss/len(test_loader))

        test_loss = running_loss / len(test_loader)
        test_acc = 100 * correct_test / total_test
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Save the trained model and plot history
    torch.save(model.state_dict(), os.path.join(OUTPUT_FOLDER, args.model_name))
    print(f"\nModel saved to {args.model_name}")
    plot_training_history(OUTPUT_FOLDER, history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an RCNN model on the SVHN dataset.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate for Adam optimizer.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training.')
    parser.add_argument('--model-name', type=str, default='rcnn_svhn.pth', help='Model name for the output.')
    parser.add_argument('--output-folder', type=str, default='output', help="Output folder")
    args = parser.parse_args()
    train(args)
