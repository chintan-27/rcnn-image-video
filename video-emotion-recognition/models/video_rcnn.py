# models/video_rcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rcl_blocks import RCL_block_3D, EfficientRCL_block_3D

class VideoRCNN_Base(nn.Module):
    """
    Shared 3D backbone.
    Provides:
      - forward_featmap(x)           -> (B,256,T,H,W)
      - forward_clip_embed(x)        -> (B,256)
      - forward_spatial_only_pool(x) -> (B,256,T)
    """
    def __init__(self, num_frames=10, efficient=False):
        super(VideoRCNN_Base, self).__init__()
        self.num_frames = num_frames
        RCL_Block = EfficientRCL_block_3D if efficient else RCL_block_3D

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3))
        self.bn1 = nn.BatchNorm3d(64)

        self.rconv1 = RCL_Block(64, 96, kernel_size=(3,3,3), t_steps=2)
        self.dropout1 = nn.Dropout3d(0.1)

        self.rconv2 = RCL_Block(96, 128, kernel_size=(3,3,3), t_steps=3)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.dropout2 = nn.Dropout3d(0.2)

        self.rconv3 = RCL_Block(128, 192, kernel_size=(3,3,3), t_steps=3)
        self.dropout3 = nn.Dropout3d(0.2)

        self.rconv4 = RCL_Block(192, 256, kernel_size=(5,3,3), t_steps=4)

        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.flatten = nn.Flatten()
        self.feature_dim = 256

    def forward_featmap(self, x):
        # x: (B,3,T,H,W) -> (B,256,T',H',W')  [T'≈T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.rconv1(x); x = self.dropout1(x)
        x = self.rconv2(x); x = self.pool1(x); x = self.dropout2(x)
        x = self.rconv3(x); x = self.dropout3(x)
        x = self.rconv4(x)
        return x

    def forward_clip_embed(self, x):
        # (B,3,T,H,W) -> (B,256)
        f = self.forward_featmap(x)
        f = self.global_pool(f)
        return self.flatten(f)

    def forward_spatial_only_pool(self, x):
        # (B,3,T,H,W) -> spatial GAP -> (B,256,T)
        f = self.forward_featmap(x)       # (B,256,T,H,W)
        f = f.mean(dim=(3,4))             # (B,256,T)
        return f


class VideoRCNN_VA(nn.Module):
    """
    VA-only model (framewise by default).
    Outputs:
      {'valence': (B,T) in [-1,1], 'arousal': (B,T) in [0,1]}
    """
    def __init__(self, num_frames=10, efficient=False):
        super(VideoRCNN_VA, self).__init__()
        self.backbone = VideoRCNN_Base(num_frames, efficient)
        # time-distributed heads: (B,256,T) -> (B,T)
        self.val_h = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
        self.aro_h = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))

    def forward(self, x):
        ft = self.backbone.forward_spatial_only_pool(x)  # (B,256,T)
        B,C,T = ft.shape
        ft2 = ft.permute(0,2,1).reshape(B*T, C)
        v = torch.tanh(self.val_h(ft2)).reshape(B,T)      # [-1,1]
        a = torch.sigmoid(self.aro_h(ft2)).reshape(B,T)   # [0,1]
        return {'valence': v, 'arousal': a}


class VideoRCNN_Emotions(nn.Module):
    """
    Emotions-only model (clip-level).
    Outputs:
      {'emotions': (B,num_emotions)}
    """
    def __init__(self, num_frames=10, num_emotions=50, efficient=False):
        super(VideoRCNN_Emotions, self).__init__()
        self.backbone = VideoRCNN_Base(num_frames, efficient)
        self.num_emotions = num_emotions
        self.emotion_head = nn.Sequential(
            nn.Linear(256,512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512,256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256,num_emotions)
        )

    def forward(self, x):
        clip = self.backbone.forward_clip_embed(x)   # (B,256)
        emo = self.emotion_head(clip)                # (B,num_emotions)
        return {'emotions': emo}


class VideoRCNN_MultiTask(nn.Module):
    """
    Multitask: framewise VA + clip-level emotions.
    Outputs:
      {
        'valence': (B,T) in [-1,1],
        'arousal': (B,T) in [0,1],
        'emotions': (B,num_emotions),
        'features': (B,256)   # clip embedding for analysis
      }
    """
    def __init__(self, num_frames=10, num_emotions=50, efficient=False):
        super(VideoRCNN_MultiTask, self).__init__()
        self.backbone = VideoRCNN_Base(num_frames, efficient)
        self.num_emotions = num_emotions

        # Emotions use clip-level features
        self.shared_features = nn.Sequential(
            nn.Linear(256,256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3)
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(256,256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.4),
            nn.Linear(256,num_emotions)
        )

        # Framewise VA heads
        self.val_h = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))
        self.aro_h = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,1))

    def forward(self, x):
        # VA path (time-preserving)
        ft = self.backbone.forward_spatial_only_pool(x)  # (B,256,T)
        B,C,T = ft.shape
        ft2 = ft.permute(0,2,1).reshape(B*T, C)
        v = torch.tanh(self.val_h(ft2)).reshape(B,T)
        a = torch.sigmoid(self.aro_h(ft2)).reshape(B,T)

        # Emotions path (clip-level)
        clip = self.backbone.forward_clip_embed(x)       # (B,256)
        shared = self.shared_features(clip)              # (B,256)
        emo = self.emotion_head(shared)                  # (B,num_emotions)

        return {'valence': v, 'arousal': a, 'emotions': emo, 'features': shared}


def get_model(model_type='multitask', num_frames=10, num_emotions=50, efficient=False):
    if model_type == 'va_only':
        return VideoRCNN_VA(num_frames, efficient)
    elif model_type == 'emotions_only':
        return VideoRCNN_Emotions(num_frames, num_emotions, efficient)
    elif model_type == 'multitask':
        return VideoRCNN_MultiTask(num_frames, num_emotions, efficient)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

