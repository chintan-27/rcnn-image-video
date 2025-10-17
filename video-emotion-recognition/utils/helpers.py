import torch
import numpy as np
import cv2
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def visualize_predictions(model, val_loader, device, num_samples=8, save_path=None):
    """Visualize model predictions on sample videos"""
    model.eval()
    
    with torch.no_grad():
        for videos, targets in val_loader:
            videos = videos.to(device)
            videos_input = videos.permute(0, 2, 1, 3, 4)  # Reshape for model
            
            predictions = model(videos_input)
            
            # Take first batch
            batch_size = min(num_samples, videos.size(0))
            videos = videos[:batch_size]
            
            fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4*batch_size))
            if batch_size == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(batch_size):
                # Show 5 frames from the video
                for j in range(5):
                    frame_idx = j * 2  # Every other frame
                    frame = videos[i, frame_idx].permute(1, 2, 0).cpu().numpy()
                    
                    # Denormalize if needed
                    frame = np.clip(frame, 0, 1)
                    
                    axes[i, j].imshow(frame)
                    axes[i, j].axis('off')
                    
                    if j == 0:  # Add prediction text on first frame
                        pred_text = []
                        if 'valence' in predictions:
                            val_pred = predictions['valence'][i].item()
                            val_true = targets['valence'][i].item()
                            pred_text.append(f"V: {val_pred:.2f} ({val_true:.2f})")
                        
                        if 'arousal' in predictions:
                            aro_pred = predictions['arousal'][i].item()
                            aro_true = targets['arousal'][i].item()
                            pred_text.append(f"A: {aro_pred:.2f} ({aro_true:.2f})")
                        
                        if 'emotions' in predictions:
                            emo_pred = predictions['emotions'][i].argmax().item()
                            emo_true = targets['emotions'][i].item()
                            pred_text.append(f"E: {emo_pred} ({emo_true})")
                        
                        axes[i, j].set_title('\n'.join(pred_text), fontsize=8)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
            
            plt.show()
            break  # Only visualize first batch

def analyze_model_efficiency(model, input_shape, device='cuda'):
    """Analyze model computational efficiency"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Time inference
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(dummy_input)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time  # Convert ms to fps
    
    return {
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'fps': fps,
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times)
    }

# evaluate.py - Separate evaluation script
import argparse
import torch
from models.video_rcnn import get_model
from data.dataset import create_data_loaders
from training.metrics import EmotionMetrics

def evaluate_model(checkpoint_path, test_csv, video_root, 
                  batch_size=16, device='cuda'):
    """Evaluate trained model on test set"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint.get('model_type', 'multitask')
    
    print(f"Loading {model_type} model from {checkpoint_path}")
    
    # Create data loader
    _, test_loader, dataset_info = create_data_loaders(
        train_csv=test_csv,  # Use same file for both (we only need test_loader)
        val_csv=test_csv,
        video_root=video_root,
        batch_size=batch_size,
        model_type=model_type,
        training=False  # Important: no augmentation
    )
    
    # Create model
    model = get_model(
        model_type=model_type,
        num_emotions=dataset_info['num_emotions']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluate
    metrics = EmotionMetrics(model_type)
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch_idx, (videos, targets) in enumerate(test_loader):
            videos = videos.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Reshape for model
            videos = videos.permute(0, 2, 1, 3, 4)
            
            # Predict
            predictions = model(videos)
            
            # Update metrics
            metrics.update(predictions, targets)
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Compute final metrics
    final_metrics = metrics.compute_all_metrics()
    
    print("\n🎯 Final Test Results:")
    print("=" * 50)
    
    for metric_name, metric_value in final_metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"{metric_name:25}: {metric_value:.4f}")
    
    # Generate plots
    if model_type in ['emotions_only', 'multitask']:
        metrics.plot_confusion_matrix(save_path='confusion_matrix.png')
        print("📊 Confusion matrix saved as 'confusion_matrix.png'")
    
    if model_type in ['va_only', 'multitask']:
        metrics.plot_va_scatter(save_path='va_scatter.png')
        print("📈 V/A scatter plot saved as 'va_scatter.png'")
    
    return final_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained emotion recognition model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--video_root', type=str, required=True,
                       help='Root directory containing video files')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        video_root=args.video_root,
        batch_size=args.batch_size,
        device=args.device
    )
