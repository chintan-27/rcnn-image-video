import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from .transforms import VideoTransforms

class CKVideoDataset(Dataset):
    """
    Dataset for Cowen-Keltner Emotional Videos
    Handles the specific directory structure and CSV format
    """
    
    def __init__(self, csv_file, video_root, num_frames=10, 
                 input_size=(112, 112), training=True, 
                 model_type='multitask', top_k_emotions=50):
        """
        Args:
            csv_file: Path to train/val/test CSV file
            video_root: Root directory (data/ckvideo/)
            Other args same as before
        """
        self.data = pd.read_csv(csv_file, header=None, 
                           names=["ID", "Type", "Arousal", "Valence"])
        self.video_root = video_root
        self.num_frames = num_frames
        self.training = training
        self.model_type = model_type
        
        # Determine which subdirectory based on CSV file name
        if 'train' in csv_file:
            self.video_subdir = 'ckvideo_middleframe_train'
        elif 'val' in csv_file:
            self.video_subdir = 'ckvideo_middleframe_val'
        elif 'test' in csv_file:
            self.video_subdir = 'ckvideo_middleframe_test'
        else:
            raise ValueError(f"Cannot determine video subdirectory from {csv_file}")
        
        # Setup transforms
        self.transforms = VideoTransforms(
            input_size=input_size,
            num_frames=num_frames,
            training=training
        )
        
        # Prepare emotion mappings for discrete classification
        if model_type in ['emotions_only', 'multitask']:
            self._prepare_emotion_mappings_ck(top_k_emotions)
    
    def _prepare_emotion_mappings_ck(self, top_k):
        """Prepare emotion label mappings for CK dataset"""
        # CK dataset likely has different emotion columns - adjust based on your CSV
        # First, let's see what columns we have
        print(f"Available columns in {len(self.data.columns)} total:")
        print(self.data.columns.tolist())
        
        # Skip non-emotion columns (adjust based on your actual CSV structure)
        skip_columns = ['Filename', 'valence', 'arousal', 'ID', 'Type']  # Add more as needed
        emotion_columns = [col for col in self.data.columns 
                          if col not in skip_columns]
        
        # For CK dataset, emotions might be binary or continuous
        # Count emotion frequencies and select top-k
        if len(emotion_columns) > 0:
            emotion_counts = self.data[emotion_columns].sum().sort_values(ascending=False)
            self.top_emotions = emotion_counts.head(top_k).index.tolist()
            
            # Create label mapping
            self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.top_emotions)}
            self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
            
            print(f"Using top {len(self.top_emotions)} emotions from CK dataset:")
            for i, emotion in enumerate(self.top_emotions[:10]):
                count = emotion_counts[emotion] if emotion in emotion_counts else 0
                print(f"  {i}: {emotion} ({count} samples)")
            if len(self.top_emotions) > 10:
                print(f"  ... and {len(self.top_emotions)-10} more")
        else:
            print("No emotion columns found, using placeholder")
            self.top_emotions = ['neutral']
            self.emotion_to_idx = {'neutral': 0}
            self.idx_to_emotion = {0: 'neutral'}
    
#    def _load_video_from_images(self, video_id):
#        """
#        Load video frames from image directory
#        CK dataset appears to have extracted frames, not video files
#        """
#        # Construct path to image directory for this video
#        video_dir = os.path.join(self.video_root, self.video_subdir, str(video_id))
#        
#        if not os.path.exists(video_dir):
#            # Try alternative naming conventions
#            alt_paths = [
#                os.path.join(self.video_root, self.video_subdir, f"video_{video_id}"),
#                os.path.join(self.video_root, self.video_subdir, f"{video_id:04d}"),
#                os.path.join(self.video_root, self.video_subdir, f"vid_{video_id}"),
#            ]
#            
#            for alt_path in alt_paths:
#                if os.path.exists(alt_path):
#                    video_dir = alt_path
#                    break
#            else:
#                raise ValueError(f"Could not find video directory for ID {video_id}")
#        
#        # Load all images from directory
#        image_files = []
#        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
#            image_files.extend(sorted([f for f in os.listdir(video_dir) if f.lower().endswith(ext)]))
#        
#        if len(image_files) == 0:
#            raise ValueError(f"No image files found in {video_dir}")
#        
#        # Load images
#        frames = []
#        for img_file in sorted(image_files):
#            img_path = os.path.join(video_dir, img_file)
#            frame = cv2.imread(img_path)
#            if frame is not None:
#                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                frames.append(frame)
#        
#        if len(frames) == 0:
#            raise ValueError(f"Could not load any frames from {video_dir}")
#        
#        return frames


    def _load_video_from_images(self, video_id):
        """
        Load video frames from flat image directory structure
        CK dataset has images as: data/ckvideo/<video_id>_<frame>.jpg
        """
        frames = []
        
        # Images are in flat directory: data/ckvideo/video_id_001.jpg to video_id_010.jpg
        base_dir = self.video_root  # data/ckvideo/
        
        for frame_idx in range(1, 11):  # Frames 1-10
            frame_filename = f"{video_id}_{frame_idx:03d}.jpg"
            img_path = os.path.join(base_dir, frame_filename)
            
            if not os.path.exists(img_path):
                raise ValueError(f"Frame not found: {img_path}")
            
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        if len(frames) == 0:
            raise ValueError(f"Could not load any frames for video {video_id}")
        
        return frames
    
    def _get_emotion_label(self, row):
        """Get dominant emotion label for discrete classification"""
        if not hasattr(self, 'top_emotions') or len(self.top_emotions) == 0:
            return 0
        
        # Check if we have emotion columns
        emotion_values = []
        for emotion in self.top_emotions:
            if emotion in row:
                emotion_values.append(row[emotion])
            else:
                emotion_values.append(0)
        
        emotion_values = pd.Series(emotion_values, index=self.top_emotions)
        
        if emotion_values.sum() == 0:
            return 0  # Default to first emotion
        
        # Get index of maximum emotion
        max_emotion = emotion_values.idxmax()
        return self.emotion_to_idx[max_emotion]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a video sample and its labels"""
        row = self.data.iloc[idx]
        
        # Get video ID (adjust column name based on your CSV)
        video_id = row.get('ID', row.get('id', row.get('video_id', idx)))
        
        try:
            frames = self._load_video_from_images(video_id)
        except Exception as e:
            print(f"Error loading video {video_id}: {e}")
            # Return a random sample instead
            return self.__getitem__((idx + 1) % len(self.data))
        
        # Apply transforms
        video_tensor = self.transforms(frames)  # (num_frames, 3, H, W)
        
        # Prepare labels based on model type
        if self.model_type == 'va_only':
            targets = {
                'valence': torch.tensor(row.get('valence', 0.0), dtype=torch.float32),
                'arousal': torch.tensor(row.get('arousal', 0.0), dtype=torch.float32)
            }
        elif self.model_type == 'emotions_only':
            targets = {
                'emotions': torch.tensor(self._get_emotion_label(row), dtype=torch.long)
            }
        else:  # multitask
            targets = {
                'valence': torch.tensor(row.get('valence', 0.0), dtype=torch.float32),
                'arousal': torch.tensor(row.get('arousal', 0.0), dtype=torch.float32),
                'emotions': torch.tensor(self._get_emotion_label(row), dtype=torch.long)
            }
        
        return video_tensor, targets

def create_ck_data_loaders(train_csv, val_csv, video_root, 
                          batch_size=8, num_workers=4, 
                          model_type='multitask', **kwargs):
    """
    Create train and validation data loaders for CK dataset
    """
    # Create datasets
    train_dataset = CKVideoDataset(
        train_csv, video_root, training=True, 
        model_type=model_type, **kwargs
    )
    
    val_dataset = CKVideoDataset(
        val_csv, video_root, training=False,
        model_type=model_type, **kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Dataset info
    dataset_info = {
        'num_emotions': len(train_dataset.top_emotions) if hasattr(train_dataset, 'top_emotions') else 0,
        'emotion_mapping': getattr(train_dataset, 'emotion_to_idx', {}),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    }
    # right after dataset_info is created
    with open(os.path.join(experiment_dir, "emotion_mapping.json"), "w") as f:
        json.dump(dataset_info.get("emotion_mapping", {}), f, indent=2)
    
    return train_loader, val_loader, dataset_info
    

create_data_loaders = create_ck_data_loaders
