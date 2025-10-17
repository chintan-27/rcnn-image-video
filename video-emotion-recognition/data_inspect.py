import pandas as pd
import os

def inspect_ck_dataset():
    """Inspect the CK dataset structure"""
    
    # Check CSV files
    for split in ['train', 'val', 'test']:
        csv_path = f'data/ckvideo/ckvideo_middleframe_{split}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\n{split.upper()} CSV ({csv_path}):")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Sample rows:")
            print(df.head(2))
            
            # Check for valence/arousal columns
            if 'valence' in df.columns:
                print(f"  Valence range: {df['valence'].min():.3f} to {df['valence'].max():.3f}")
            if 'arousal' in df.columns:
                print(f"  Arousal range: {df['arousal'].min():.3f} to {df['arousal'].max():.3f}")
        else:
            print(f"{csv_path} not found")
    
    # Check video directories
    for split in ['train', 'val', 'test']:
        video_dir = f'data/ckvideo/ckvideo_middleframe_{split}'
        if os.path.exists(video_dir):
            subdirs = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
            print(f"\n{split.upper()} Videos ({video_dir}):")
            print(f"  Number of video directories: {len(subdirs)}")
            if len(subdirs) > 0:
                sample_dir = os.path.join(video_dir, subdirs[0])
                frames = len([f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                print(f"  Sample directory: {subdirs[0]} ({frames} frames)")
                print(f"  First few directories: {subdirs[:5]}")
        else:
            print(f"{video_dir} not found")

if __name__ == '__main__':
    inspect_ck_dataset()
