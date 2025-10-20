import os
import pandas as pd
from pathlib import Path

def inspect_ck_dataset(root="data/ckvideo_out"):
    """
    Inspect the CK dataset (restructured layout).
    Displays CSV info, valence/arousal stats, frame counts, and metadata summary.
    """
    root = Path(root)
    print("=" * 60)
    print(f"Inspecting CK Dataset at: {root.resolve()}")
    print("=" * 60)

    # ------------------------------------------------
    # Check CSV files under splits/
    # ------------------------------------------------
    splits_dir = root / "splits"
    if not splits_dir.exists():
        print(f"[ERROR] Splits directory not found: {splits_dir}")
        return

    for split in ["train", "val", "test"]:
        csv_path = splits_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"[WARN] {split.upper()} CSV not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path}: {e}")
            continue

        print(f"\n{split.upper()} CSV: {csv_path}")
        print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print(f"  Columns: {', '.join(df.columns)}")

        print("  Sample rows:")
        print(df.head(2).to_string(index=False))

        # Valence/Arousal stats
        if "valence" in df.columns:
            print(f"  Valence range: {df['valence'].min():.3f} to {df['valence'].max():.3f}")
        if "arousal" in df.columns:
            print(f"  Arousal range: {df['arousal'].min():.3f} to {df['arousal'].max():.3f}")

    # ------------------------------------------------
    # Check video frame directories
    # ------------------------------------------------
    frames_dir = root / "frames"
    if not frames_dir.exists():
        print(f"\n[ERROR] Frames directory not found: {frames_dir}")
        return

    for split in ["train", "val", "test"]:
        split_dir = frames_dir / split
        if not split_dir.exists():
            print(f"\n[WARN] {split.upper()} frames directory missing: {split_dir}")
            continue

        video_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"\n{split.upper()} FRAMES: {split_dir}")
        print(f"  Number of video directories: {len(video_dirs)}")

        if not video_dirs:
            continue

        # Inspect first few video folders
        for vid_dir in video_dirs[:3]:
            frame_files = sorted([f for f in vid_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
            n_frames = len(frame_files)
            print(f"  Video ID {vid_dir.name}: {n_frames} frames")
            if n_frames != 10:
                print(f"    Warning: Expected 10 frames, found {n_frames}")
        if len(video_dirs) > 3:
            print(f"  ...and {len(video_dirs)-3} more videos")

    # ------------------------------------------------
    # Check metadata
    # ------------------------------------------------
    metadata_dir = root / "metadata"
    if metadata_dir.exists():
        print(f"\nMetadata directory: {metadata_dir}")
        for file in metadata_dir.iterdir():
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")
    else:
        print("\n[WARN] Metadata directory not found")

    print("\nInspection complete.")

if __name__ == "__main__":
    inspect_ck_dataset()
