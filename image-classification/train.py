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
    Main training function with improved regularization.
    Saves checkpoints using your provided model name (e.g., rcnn_svhn.pth):
      - <base>_best_acc.pth  for highest validation accuracy
      - <base>_best_loss.pth for lowest validation loss
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"Using device: {device}")
    
    # Make the Output Folder
    os.makedirs(args.output_folder, exist_ok=True)
    OUTPUT_FOLDER = args.output_folder
    
    # Derive checkpoint filenames from the provided model name
    base_name, ext = os.path.splitext(args.model_name)
    best_acc_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_best_acc{ext}")
    best_loss_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_best_loss{ext}")
    
    # Load data
    train_loader, test_loader = get_data_loaders(args.batch_size)
    
    # Initialize model, loss, and optimizer
    model = RCNN(num_classes=10)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ADD WEIGHT DECAY FOR REGULARIZATION
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # ADD LEARNING RATE SCHEDULER
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.scheduler_patience 
    )
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    # Track best metrics
    best_acc = -1.0
    best_loss = float("inf")
    
    # ADD EARLY STOPPING VARIABLES
    patience_counter = 0
    
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
        train_acc = 100.0 * correct_train / total_train
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
        test_acc = 100.0 * correct_test / total_test
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # STEP THE SCHEDULER
        scheduler.step(test_loss)
        
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"  # Show current learning rate
        )
        
        # ---- Checkpointing: save ONLY if metric improves ----
        improvement_made = False
        
        # (1) Best Accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            improvement_made = True
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_metric": "acc",
                "val_acc": test_acc,
                "val_loss": test_loss,
                "args": vars(args),
            }, best_acc_path)
            print(f"Saved new best-ACC model ({best_acc:.2f}%) to {best_acc_path}")
        
        # (2) Best Loss
        if test_loss < best_loss:
            best_loss = test_loss
            improvement_made = True
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_metric": "loss",
                "val_acc": test_acc,
                "val_loss": test_loss,
                "args": vars(args),
            }, best_loss_path)
            print(f"Saved new best-LOSS model ({best_loss:.4f}) to {best_loss_path}")
        
        # EARLY STOPPING LOGIC
        if improvement_made:
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
    
    print(f"Training completed. Best accuracy: {best_acc:.2f}%, Best loss: {best_loss:.4f}")
    
    # Plot history after training
    plot_training_history(OUTPUT_FOLDER, history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an RCNN model on the SVHN dataset.")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for regularization.')
    parser.add_argument('--early-stop-patience', type=int, default=20, help='Early stopping patience.')
    parser.add_argument('--scheduler-patience', type=int, default=10, help='LR scheduler patience.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training.')
    parser.add_argument('--model-name', type=str, default='rcnn_svhn.pth', help='Model name for the output.')
    parser.add_argument('--output-folder', type=str, default='output', help="Output folder")
    
    args = parser.parse_args()
    train(args)
