# main.py
import os
import argparse
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from models.video_rcnn import get_model
from models.losses import get_loss_function
from data_utils.dataset import create_data_loaders
from training.trainer import EmotionTrainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Video Emotion Recognition with 3D RCNN')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='multitask',
                       choices=['va_only', 'emotions_only', 'multitask'],
                       help='Type of model to train')
    parser.add_argument('--efficient', action='store_true',
                       help='Use efficient depthwise separable convolutions')
    parser.add_argument('--num_frames', type=int, default=10,
                       help='Number of frames per video sequence')
    parser.add_argument('--top_k_emotions', type=int, default=50,
                       help='Number of top emotions to classify')
    
    # Data configuration
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True,
                       help='Path to validation CSV file')
    parser.add_argument('--video_root', type=str, required=True,
                       help='Root directory containing video files')
    parser.add_argument('--input_size', type=int, nargs=2, default=[112, 112],
                       help='Input spatial dimensions (height width)')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Loss configuration
    parser.add_argument('--valence_weight', type=float, default=1.0,
                       help='Weight for valence loss')
    parser.add_argument('--arousal_weight', type=float, default=1.0,
                       help='Weight for arousal loss')
    parser.add_argument('--emotion_weight', type=float, default=1.0,
                       help='Weight for emotion classification loss')
    parser.add_argument('--adaptive_weights', action='store_true',
                       help='Use adaptive loss weighting')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing for emotion classification')
    
    # Training options
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                       choices=['plateau', 'step', 'none'],
                       help='Learning rate scheduler type')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (AMP)')
    
    # Paths
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Experiment naming
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment')
    
    return parser.parse_args()

def create_experiment_name(args):
    """Create experiment name based on configuration"""
    if args.experiment_name:
        return args.experiment_name
    
    name_parts = [
        args.model_type,
        f"frames{args.num_frames}",
        f"bs{args.batch_size}",
        f"lr{args.learning_rate}"
    ]
    
    if args.efficient:
        name_parts.append("efficient")
    
    if args.model_type in ['emotions_only', 'multitask']:
        name_parts.append(f"top{args.top_k_emotions}")
    
    return "_".join(name_parts)

def save_config(args, save_path):
    """Save configuration to JSON file"""
    config = vars(args).copy()
    
    # Convert non-serializable objects to strings
    for key, value in config.items():
        if not isinstance(value, (str, int, float, bool, list, type(None))):
            config[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

def create_optimizer(model, args):
    """Create optimizer based on configuration"""
    if args.weight_decay > 0:
        # Separate weight decay for different parameter types
        param_groups = [
            {'params': [p for n, p in model.named_parameters() 
                       if 'bias' not in n and 'bn' not in n], 
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() 
                       if 'bias' in n or 'bn' in n], 
             'weight_decay': 0.0}
        ]
        optimizer = optim.AdamW(param_groups, lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    return optimizer

def create_scheduler(optimizer, args):
    """Create learning rate scheduler"""
    if args.scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, 
            patience=5, verbose=True, min_lr=1e-7
        )
    elif args.scheduler_type == 'step':
        return StepLR(optimizer, step_size=20, gamma=0.5)
    elif args.scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")

def main():
    """Main training function"""
    args = parse_args()
    
    # Create experiment directory
    experiment_name = create_experiment_name(args)
    experiment_dir = os.path.join(args.save_dir, experiment_name)
    log_dir = os.path.join(args.log_dir, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    save_config(args, config_path)
    
    print(f"🚀 Starting experiment: {experiment_name}")
    print(f"📁 Experiment directory: {experiment_dir}")
    print(f"📊 Model type: {args.model_type}")
    print(f"🎥 Video frames: {args.num_frames}")
    print(f"📐 Input size: {args.input_size}")
    
    # Set device
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
        print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Create data loaders
    print("📊 Loading datasets...")
    train_loader, val_loader, dataset_info = create_data_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        video_root=args.video_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_type=args.model_type,
        num_frames=args.num_frames,
        input_size=tuple(args.input_size),
        top_k_emotions=args.top_k_emotions
    )
    
    print(f"📈 Training samples: {dataset_info['train_size']}")
    print(f"📉 Validation samples: {dataset_info['val_size']}")
    
    if args.model_type in ['emotions_only', 'multitask']:
        print(f"😊 Number of emotions: {dataset_info['num_emotions']}")
    
    # Create model
    print("🧠 Creating model...")
    model = get_model(
        model_type=args.model_type,
        num_frames=args.num_frames,
        num_emotions=dataset_info['num_emotions'],
        efficient=args.efficient
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🎯 Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_kwargs = {
        'valence_weight': args.valence_weight,
        'arousal_weight': args.arousal_weight,
        'emotion_weight': args.emotion_weight,
        'adaptive_weights': args.adaptive_weights,
        'label_smoothing': args.label_smoothing
    }
    
    criterion = get_loss_function(args.model_type, **loss_kwargs)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    
    # Get emotion names for metrics
    emotion_names = []
    if 'emotion_mapping' in dataset_info:
        emotion_names = [None] * len(dataset_info['emotion_mapping'])
        for emotion, idx in dataset_info['emotion_mapping'].items():
            emotion_names[idx] = emotion
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_type=args.model_type,
        save_dir=experiment_dir,
        log_dir=log_dir,
        emotion_names=emotion_names
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("🏋️ Starting training...")
    try:
        best_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping_patience,
            save_every=args.save_every
        )
        
        print("🎉 Training completed successfully!")
        print("🏆 Best metrics:")
        for metric_name, metric_value in best_metrics.items():
            if isinstance(metric_value, (int, float)):
                print(f"  {metric_name}: {metric_value:.4f}")
        
        # Save final results
        results_path = os.path.join(experiment_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for k, v in best_metrics.items():
                if isinstance(v, (list, tuple)):
                    serializable_metrics[k] = [float(x) if hasattr(x, 'item') else x for x in v]
                elif hasattr(v, 'item'):
                    serializable_metrics[k] = v.item()
                else:
                    serializable_metrics[k] = v
            
            json.dump(serializable_metrics, f, indent=2)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.save_checkpoint({}, is_best=False)
        print("💾 Checkpoint saved")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        trainer.save_checkpoint({}, is_best=False)
        print("💾 Emergency checkpoint saved")
        raise

if __name__ == '__main__':
    main()
