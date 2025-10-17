import torch
import cv2
import numpy as np
from torchvision import transforms
import random

class VideoTransforms:
    """Video-specific data augmentation and preprocessing"""
    
    def __init__(self, input_size=(112, 112), num_frames=10, 
                 training=True, normalize=True):
        self.input_size = input_size
        self.num_frames = num_frames
        self.training = training
        
        # Normalization values (ImageNet pretrained)
        if normalize:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = None
    
    def temporal_crop(self, frames, num_frames):
        """Extract num_frames from video sequence"""
        total_frames = len(frames)
        
        if total_frames <= num_frames:
            # Repeat last frame if video is too short
            frames = frames + [frames[-1]] * (num_frames - total_frames)
        else:
            if self.training:
                # Random temporal crop during training
                start_idx = random.randint(0, total_frames - num_frames)
            else:
                # Center crop during validation/testing
                start_idx = (total_frames - num_frames) // 2
            
            frames = frames[start_idx:start_idx + num_frames]
        
        return frames
    
    def spatial_transforms(self, frames):
        """Apply spatial augmentations"""
        transformed_frames = []
        
        for frame in frames:
            # Convert to PIL if needed
            if isinstance(frame, np.ndarray):
                frame = transforms.ToPILImage()(frame)
            
            # Spatial augmentations for training
            if self.training:
                # Random resized crop
                frame = transforms.RandomResizedCrop(
                    self.input_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1)
                )(frame)
                
                # Random horizontal flip
                if random.random() > 0.5:
                    frame = transforms.RandomHorizontalFlip(p=1.0)(frame)
                
                # Color jittering
                frame = transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )(frame)
            else:
                # Simple resize for validation/testing
                frame = transforms.Resize(self.input_size)(frame)
                frame = transforms.CenterCrop(self.input_size)(frame)
            
            # Convert to tensor
            frame = transforms.ToTensor()(frame)
            
            # Normalize if specified
            if self.normalize:
                frame = self.normalize(frame)
            
            transformed_frames.append(frame)
        
        return transformed_frames
    
    def temporal_augmentation(self, frames):
        """Apply temporal augmentations"""
        if not self.training:
            return frames
        
        # Random temporal reversal (for some emotions)
        if random.random() > 0.9:
            frames = frames[::-1]
        
        # Random frame dropout (simulate missing frames)
        if random.random() > 0.95:
            dropout_idx = random.randint(0, len(frames) - 1)
            frames[dropout_idx] = frames[max(0, dropout_idx - 1)]
        
        return frames
    
    def __call__(self, video_frames):
        """
        Apply all transformations to video sequence
        
        Args:
            video_frames: List of frames or numpy array
            
        Returns:
            Tensor of shape (num_frames, 3, height, width)
        """
        # Temporal cropping
        frames = self.temporal_crop(video_frames, self.num_frames)
        
        # Temporal augmentation
        frames = self.temporal_augmentation(frames)
        
        # Spatial transformations
        frames = self.spatial_transforms(frames)
        
        # Stack frames into tensor
        video_tensor = torch.stack(frames)  # (num_frames, 3, H, W)
        
        return video_tensor

