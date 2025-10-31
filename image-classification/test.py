import os
import torch
import argparse
from tqdm import tqdm

from model import RCNN
from utils import get_data_loaders, plot_predictions

def test(args):
    """
    Main testing function.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    
    # Safer output folder name from the model filename
    model_basename = os.path.splitext(os.path.basename(args.model_path))[0]
    output_folder = os.path.join(args.data_folder, model_basename)
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Using device: {device}")
    
    # Load test data
    _, test_loader = get_data_loaders(args.batch_size)
    
    # Initialize model
    model = RCNN(num_classes=10).to(device)
    
    # Load checkpoint or raw state_dict robustly
    try:
        model_path = os.path.join(args.data_folder, args.model_path)
        ckpt = torch.load(model_path, map_location=device)
    
        # Accept either a checkpoint dict or a bare state_dict
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    
        # Strip 'module.' if saved from DataParallel
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    
        model.load_state_dict(state, strict=True)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please run train.py first or provide the correct --model-path.")
        return 
    model.eval()

    correct_test = 0
    total_test = 0
    
    print("Evaluating model on the test set...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    final_acc = 100 * correct_test / total_test
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")

    # Plot some sample predictions
    print("Generating sample prediction plot...")
    plot_predictions(model, test_loader, device, output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a trained RCNN model on the SVHN dataset.")
    parser.add_argument('--model-path', type=str, default='rcnn_svhn.pth', help='Path to the trained model file.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for testing.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for testing.')
    parser.add_argument('--data-folder', type=str, default='data', help='Model Folder and output folder')    
    args = parser.parse_args()
    test(args)

