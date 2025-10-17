# models/video_rcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rcl_blocks import RCL_block_3D, EfficientRCL_block_3D

class VideoRCNN_Base(nn.Module):
    """
    Shared backbone for all three training approaches (A, B, C)
    Extracts rich spatiotemporal features from video sequences
    """
    def __init__(self, num_frames=10, efficient=False):
        super(VideoRCNN_Base, self).__init__()
        
        self.num_frames = num_frames
        
        # Choose block type based on efficiency needs
        RCL_Block = EfficientRCL_block_3D if efficient else RCL_block_3D
        
        # Initial 3D convolution - from RGB to initial features
        self.conv1 = nn.Conv3d(
            3, 64, 
            kernel_size=(3, 7, 7),  # Larger spatial kernel for initial features
            stride=(1, 2, 2),       # Downsample spatially, keep temporal
            padding=(1, 3, 3)
        )
        self.bn1 = nn.BatchNorm3d(64)
        
        # Progressive RCL blocks - increasing complexity
        self.rconv1 = RCL_Block(64, 96, kernel_size=(3, 3, 3), t_steps=2)
        self.dropout1 = nn.Dropout3d(0.1)
        
        self.rconv2 = RCL_Block(96, 128, kernel_size=(3, 3, 3), t_steps=3)
        
        # Spatiotemporal pooling - reduce dimensions while preserving info
        self.pool1 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), 
            stride=(2, 2, 2)
        )
        self.dropout2 = nn.Dropout3d(0.2)
        
        self.rconv3 = RCL_Block(128, 192, kernel_size=(3, 3, 3), t_steps=3)
        self.dropout3 = nn.Dropout3d(0.2)
        
        # Final recurrent processing with larger temporal context
        self.rconv4 = RCL_Block(192, 256, kernel_size=(5, 3, 3), t_steps=4)
        
        # Global spatiotemporal pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        
        # Feature dimension for downstream heads
        self.feature_dim = 256
    
    def forward(self, x):
        """
        Extract spatiotemporal features from video
        
        Args:
            x: Input video tensor (batch, 3, frames, height, width)
            
        Returns:
            Feature vector (batch, feature_dim)
        """
        # Initial processing
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Recurrent feature extraction and refinement
        x = self.rconv1(x)
        x = self.dropout1(x)
        
        x = self.rconv2(x)
        x = self.pool1(x)
        x = self.dropout2(x)
        
        x = self.rconv3(x)
        x = self.dropout3(x)
        
        x = self.rconv4(x)
        
        # Global feature extraction
        features = self.global_pool(x)
        features = self.flatten(features)
        
        return features

class VideoRCNN_VA(nn.Module):
    """Option A: Valence/Arousal Only Model"""
    def __init__(self, num_frames=10, efficient=False):
        super(VideoRCNN_VA, self).__init__()
        
        self.backbone = VideoRCNN_Base(num_frames, efficient)
        
        # Valence head: predicts emotional positivity/negativity [-1, 1]
        self.valence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Arousal head: predicts emotional intensity [0, 1]
        self.arousal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        valence = self.valence_head(features).squeeze(-1)
        arousal = self.arousal_head(features).squeeze(-1)
        
        return {
            'valence': valence,
            'arousal': arousal
        }

class VideoRCNN_Emotions(nn.Module):
    """Option B: Discrete Emotions Classification"""
    def __init__(self, num_frames=10, num_emotions=50, efficient=False):
        super(VideoRCNN_Emotions, self).__init__()
        
        self.backbone = VideoRCNN_Base(num_frames, efficient)
        self.num_emotions = num_emotions
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        emotions = self.emotion_head(features)
        
        return {'emotions': emotions}

class VideoRCNN_MultiTask(nn.Module):
    """Option C: Multi-Task Learning - The Complete System!"""
    def __init__(self, num_frames=10, num_emotions=50, efficient=False):
        super(VideoRCNN_MultiTask, self).__init__()
        
        self.backbone = VideoRCNN_Base(num_frames, efficient)
        self.num_emotions = num_emotions
        
        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        
        # Valence prediction head
        self.valence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        # Arousal prediction head
        self.arousal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Discrete emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_emotions)
        )
    
    def forward(self, x):
        # Extract shared features
        backbone_features = self.backbone(x)
        shared_features = self.shared_features(backbone_features)
        
        # Task-specific predictions
        valence = self.valence_head(shared_features).squeeze(-1)
        arousal = self.arousal_head(shared_features).squeeze(-1)
        emotions = self.emotion_head(shared_features)
        
        return {
            'valence': valence,
            'arousal': arousal,
            'emotions': emotions,
            'features': shared_features  # For analysis
        }

def get_model(model_type='multitask', num_frames=10, num_emotions=50, efficient=False):
    """
    Factory function to create models
    
    Args:
        model_type: 'va_only', 'emotions_only', or 'multitask'
        num_frames: Number of input frames
        num_emotions: Number of discrete emotion classes
        efficient: Use efficient depthwise separable convolutions
    
    Returns:
        Model instance
    """
    if model_type == 'va_only':
        return VideoRCNN_VA(num_frames, efficient)
    elif model_type == 'emotions_only':
        return VideoRCNN_Emotions(num_frames, num_emotions, efficient)
    elif model_type == 'multitask':
        return VideoRCNN_MultiTask(num_frames, num_emotions, efficient)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
