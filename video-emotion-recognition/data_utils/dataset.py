import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from .transforms import VideoTransforms


class CKVideoDataset(Dataset):
    """
    Dataset for Cowen-Keltner Emotional Videos (restructured layout)
    - Accepts split CSVs where each row is a *frame ID* like '2177_001'
    - Loads frames from: <video_root>/frames/<split>/<VIDEO_ID>/<FRAME>.jpg
    - Returns per-frame VA sequences (T,) aligned to sampled frames
    - Optionally joins video-level emotion columns from metadata CSV
    """

    def __init__(
        self,
        csv_file,
        video_root,
        num_frames=10,
        input_size=(112, 112),
        training=True,
        model_type='multitask',
        top_k_emotions=50,
        emotion_mapping=None,
        emotion_names=None
    ):
        """
        Args:
            csv_file: Path to split CSV (train/val/test) with columns ID,Type,Arousal,Valence
            video_root: Root directory for restructured dataset (ckvideo_out/)
            num_frames: number of frames sampled per clip
            input_size: (H, W) for spatial transforms
            training: whether to apply train-time augmentation
            model_type: 'va_only' | 'emotions_only' | 'multitask'
            top_k_emotions: number of discrete emotions to keep (ranked by frequency)
        """
        # ---- Load CSV & normalize columns ----
        self.data = pd.read_csv(csv_file)
        self.data.columns = [c.lower() for c in self.data.columns]

        if 'id' not in self.data.columns:
            raise ValueError(
                f"{csv_file} must contain an 'ID' column (e.g., frame id '2177_001')."
            )

        # Add base_id for grouping frames into a clip (video-level) and frame key
        self.data['base_id'] = self.data['id'].astype(str).str.split('_').str[0]
        self.data['frame'] = self.data['id'].astype(str).str.split('_').str[1].str.zfill(3)

        self.video_root = video_root
        self.num_frames = num_frames
        self.training = training
        self.model_type = model_type
        self.emotion_mapping_override = emotion_mapping
        self.emotion_names_override   = emotion_names

        # ---- Determine split from CSV filename ----
        csv_name = os.path.basename(csv_file).lower()
        if 'train' in csv_name:
            self.video_subdir = os.path.join('frames', 'train')
        elif 'val' in csv_name:
            self.video_subdir = os.path.join('frames', 'val')
        elif 'test' in csv_name:
            self.video_subdir = os.path.join('frames', 'test')
        else:
            raise ValueError(f"Cannot determine split from {csv_file}")

        # ---- Transforms ----
        self.transforms = VideoTransforms(
            input_size=input_size,
            num_frames=num_frames,
            training=training,
        )

        # ---- Build per-frame VA map -> { base_id: { '001': (v,a), ... } } ----
        self.frame_va = {}
        has_va = all(col in self.data.columns for col in ['valence', 'arousal'])
        if has_va:
            for (bid, f, v, a) in self.data[['base_id', 'frame', 'valence', 'arousal']].itertuples(index=False, name=None):
                self.frame_va.setdefault(str(bid), {})[str(f)] = (float(v), float(a))

        # ---- Emotions: ensure emotion columns are available when requested ----
        if model_type in ['emotions_only', 'multitask']:
            self._maybe_join_emotions_from_metadata()
            self._prepare_emotion_mappings_ck(top_k_emotions)

    # ---------------------------- Emotions utilities ----------------------------

    def _maybe_join_emotions_from_metadata(self):
        """
        If split CSV lacks discrete emotion columns, try to join from:
          <video_root>/metadata/CowenKeltnerEmotionalVideos.csv
        We match by base_id = Filename without '.mp4'
        """
        # Heuristic: if we already have many non-VA columns, assume emotions are present.
        skip = set(['filename', 'valence', 'arousal', 'id', 'type', 'base_id', 'frame'])
        current_emotion_cols = [c for c in self.data.columns if c not in skip]
        if len(current_emotion_cols) > 4:
            return  # already have emotion columns

        meta_path = os.path.join(self.video_root, 'metadata', 'CowenKeltnerEmotionalVideos.csv')
        if not os.path.isfile(meta_path):
            # Nothing to join; _prepare_emotion_mappings_ck will raise if required
            return

        meta = pd.read_csv(meta_path, low_memory=False)
        # Standardize 'base_id'
        if 'Filename' in meta.columns:
            meta['base_id'] = meta['Filename'].astype(str).str.replace('.mp4', '', regex=False)
        elif 'filename' in meta.columns:
            meta['base_id'] = meta['filename'].astype(str).str.replace('.mp4', '', regex=False)
        else:
            # Can't align — no recognized filename column
            return

        # Drop obvious non-emotion columns if present
        drop_like = {'filename', 'Filename', 'valence', 'Valence', 'arousal', 'Arousal', 'base_id'}
        meta_cols = [c for c in meta.columns if c not in drop_like]

        # Left-join meta emotion columns onto EVERY split row (we index by base_id later)
        self.data = self.data.merge(meta[['base_id'] + meta_cols], on='base_id', how='left')

    def _prepare_emotion_mappings_ck(self, top_k):
        """Prepare discrete emotion label mappings from columns."""
        print(f"Available columns in {len(self.data.columns)} total:")
        print(self.data.columns.tolist())

        # Skip non-emotion columns
        skip_columns = ['filename', 'valence', 'arousal', 'id', 'type', 'base_id', 'frame']
        emotion_columns = [col for col in self.data.columns if col not in skip_columns]

        if not emotion_columns:
            raise ValueError(
                "No discrete emotion columns found in split CSV. "
                "Either join with metadata/CowenKeltnerEmotionalVideos.csv or use model_type='va_only'."
            )

        # Emotion counts (assume numeric/binary/weights) → pick top-K
        # Robustness: coerce to numeric; NaNs → 0
        emo_df = self.data[emotion_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        emotion_counts = emo_df.sum().sort_values(ascending=False)
        self.top_emotions = emotion_counts.head(top_k).index.tolist()

        # Mapping dicts
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.top_emotions)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}

        print(f"Using top {len(self.top_emotions)} emotions:")
        for i, emotion in enumerate(self.top_emotions[:10]):
            count = float(emotion_counts.get(emotion, 0.0))
            print(f"  {i}: {emotion} ({count} samples)")
        if len(self.top_emotions) > 10:
            print(f"  ... and {len(self.top_emotions)-10} more")

    def _get_emotion_label(self, row):
        """Return dominant emotion index for a given row (clip)."""
        if not hasattr(self, 'top_emotions') or len(self.top_emotions) == 0:
            return 0

        values = []
        for emotion in self.top_emotions:
            values.append(row.get(emotion, 0))
        s = pd.Series(values, index=self.top_emotions)
        if s.sum() == 0:
            return 0
        max_emotion = s.idxmax()
        return self.emotion_to_idx[max_emotion]

    # ---------------------------- Frame loading ----------------------------

    def _load_video_from_images(self, video_id):
        """
        Load frames from: <video_root>/<frames>/<split>/<VIDEO_ID>/<FRAME>.jpg
        Returns: (list_of_images_RGB, list_of_frame_keys)
        """
        video_dir = os.path.join(self.video_root, self.video_subdir, str(video_id))
        if not os.path.isdir(video_dir):
            raise ValueError(f"Could not find video directory for ID {video_id} at {video_dir}")

        # Collect candidate frames (.jpg)
        all_frames = sorted([f for f in os.listdir(video_dir) if f.lower().endswith('.jpg')])
        if not all_frames:
            raise ValueError(f"No image files found in {video_dir}")

        # Sample/pad to self.num_frames
        if len(all_frames) >= self.num_frames:
            # uniform subsample
            step = max(1, len(all_frames) // self.num_frames)
            chosen = all_frames[::step][:self.num_frames]
        else:
            # Not enough frames: take all and pad last
            chosen = all_frames + [all_frames[-1]] * (self.num_frames - len(all_frames))

        frames = []
        chosen_keys = []
        for fname in chosen:
            img_path = os.path.join(video_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            chosen_keys.append(os.path.splitext(fname)[0])

        if not frames:
            raise ValueError(f"Could not load any frames from {video_dir}")

        return frames, chosen_keys

    # ---------------------------- Torch Dataset API ----------------------------

    def __len__(self):
        # Treat each *row* as a candidate sample, but we will normalize by base_id in __getitem__
        return len(self.data)

    def __getitem__(self, idx):
        # Try a few times to avoid bad samples
        for _ in range(3):
            row = self.data.iloc[idx]
    
            # Normalize to base (video) id so we read the clip folder
            raw_id = str(row.get('id', row.get('ID', row.get('video_id', idx))))
            video_id = raw_id.split('_')[0]  # "2177_001" -> "2177"
    
            try:
                # Load frames and apply transforms
                frames_np, chosen_keys = self._load_video_from_images(video_id)
                video_tensor = self.transforms(frames_np)  # (T, C, H, W)
    
                # Build framewise VA sequences aligned to chosen_keys
                T = video_tensor.shape[0]
                v_seq = np.zeros((T,), dtype=np.float32)
                a_seq = np.zeros((T,), dtype=np.float32)
    
                if video_id in self.frame_va:
                    for i, k in enumerate(chosen_keys[:T]):
                        if k in self.frame_va[video_id]:
                            v_seq[i], a_seq[i] = self.frame_va[video_id][k]
                        else:
                            v_seq[i] = float(row.get('valence', 0.0))
                            a_seq[i] = float(row.get('arousal', 0.0))
                else:
                    v_seq[:] = float(row.get('valence', 0.0))
                    a_seq[:] = float(row.get('arousal', 0.0))
    
                # Normalize to model ranges
                v_seq = (v_seq - 5.0) / 4.0   # 1..9 -> -1..1
                a_seq = a_seq / 9.0           # 0..9 -> 0..1
                v_seq = np.clip(v_seq, -1.0, 1.0)
                a_seq = np.clip(a_seq,  0.0, 1.0)
    
                # Build targets
                if self.model_type == 'va_only':
                    targets = {
                        'valence': torch.from_numpy(v_seq),   # (T,)
                        'arousal': torch.from_numpy(a_seq),   # (T,)
                    }
                elif self.model_type == 'emotions_only':
                    targets = {
                        'emotions': torch.tensor(self._get_emotion_label(row), dtype=torch.long)
                    }
                else:  # multitask
                    targets = {
                        'valence': torch.from_numpy(v_seq),
                        'arousal': torch.from_numpy(a_seq),
                        'emotions': torch.tensor(self._get_emotion_label(row), dtype=torch.long),
                    }
    
                return video_tensor, targets
    
            except Exception as e:
                # On failure, move to next row
                print(f"Error loading video {video_id}: {e}")
                idx = (idx + 1) % len(self.data)
    
        # If repeated failures occur, raise so the calling code can surface the issue
        raise RuntimeError("Failed to load a valid sample after several attempts.")
    

def create_ck_data_loaders(
    train_csv,
    val_csv,
    video_root,
    batch_size=8,
    num_workers=4,
    model_type='multitask',
    experiment_dir=None,
    **kwargs,
):
    """
    Create train and validation data loaders for CK dataset (restructured)
    """
    train_dataset = CKVideoDataset(
        train_csv, video_root, training=True, model_type=model_type, **kwargs
    )

    # Extract mapping from train
    emotion_mapping = getattr(train_dataset, 'emotion_to_idx', None)
    emotion_names = None
    if emotion_mapping:
        # stable idx->name list
        emotion_names = [e for e, i in sorted(emotion_mapping.items(), key=lambda x: x[1])]

    # VAL uses the SAME mapping
    val_dataset = CKVideoDataset(
        val_csv, video_root, training=False, model_type=model_type,
        emotion_mapping=emotion_mapping, emotion_names=emotion_names, **kwargs
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    dataset_info = {
        'num_emotions': len(getattr(train_dataset, 'top_emotions', [])),
        'emotion_mapping': emotion_mapping or {},
        'emotion_names': emotion_names or [],
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
    }

    if experiment_dir:
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'emotion_mapping.json'), 'w') as f:
            json.dump(dataset_info['emotion_mapping'], f, indent=2)

    return train_loader, val_loader, dataset_info

# Backwards-compat alias
create_data_loaders = create_ck_data_loaders

