# debug/debug_step2_batch.py
import os, sys, json, random, argparse, time
import numpy as np
import torch
from torch import nn

# your project imports
from data_utils.dataset import create_ck_data_loaders
from models.video_rcnn import get_model


def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tstats(t, name):
    t_cpu = t.detach().float().cpu()
    return f"{name} shape={tuple(t.shape)} min={t_cpu.min():.3f} max={t_cpu.max():.3f} mean={t_cpu.mean():.3f} std={t_cpu.std():.3f}"


def main(args):
    print(">>> Setting seed...")
    set_seed(args.seed)

    print(">>> Checking CUDA availability...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n>>> Building data loaders...")
    start = time.time()
    train_loader, val_loader, info = create_ck_data_loaders(
        train_csv=os.path.join(args.data_root, "splits", "train.csv"),
        val_csv=os.path.join(args.data_root, "splits", "val.csv"),
        video_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_type=args.model_type,
        experiment_dir=None,
        num_frames=args.num_frames,
        input_size=(args.height, args.width),
        top_k_emotions=args.num_emotions
    )
    print(f"Data loaders built in {time.time() - start:.2f}s")

    print("\n=== DATASET INFO ===")
    print(json.dumps(info, indent=2)[:2000])

    print("\n>>> Fetching one batch from train loader...")
    videos, targets = next(iter(train_loader))
    print(f"Loaded batch of {videos.shape[0]} videos")

    print("videos shape:", videos.shape)
    for k, v in targets.items():
        print(f"targets[{k}] shape={tuple(v.shape)} dtype={v.dtype}")

    print("\n>>> Moving batch to device and permuting (B, T, C, H, W) → (B, C, T, H, W)...")
    videos = videos.to(device).permute(0, 2, 1, 3, 4).contiguous()
    targets = {k: v.to(device) for k, v in targets.items()}
    print("videos moved to:", videos.device)

    print("\n>>> Building model...")
    model = get_model(
        model_type=args.model_type,
        num_frames=args.num_frames,
        num_emotions=info.get("num_emotions", 50),
        efficient=args.efficient,
    ).to(device)
    model.eval()
    print("Model initialized and moved to:", next(model.parameters()).device)

    print("\n>>> Forward pass...")
    with torch.no_grad():
        preds = model(videos)
    print("Forward pass done.")

    print("\n=== MODEL OUTPUT STATS ===")
    for k, v in preds.items():
        print(tstats(v, f"preds[{k}]"))

    if "valence" in preds and "arousal" in preds:
        pv, pa = preds["valence"], preds["arousal"]
        tv, ta = targets["valence"], targets["arousal"]

        assert pv.dim() in (1, 2)
        print("\n>>> Checking VA alignment...")
        if pv.dim() == 2:
            print(f"VA outputs are sequences (B,T) = {pv.shape}")
        else:
            print(f"VA outputs are clip-level (B,) = {pv.shape}")

        print(tstats(tv, "targets[valence]"))
        print(tstats(ta, "targets[arousal]"))

        print("\n>>> First 10 frame samples from batch[0]:")
        T = min(10, pv.shape[-1] if pv.dim() == 2 else 1)
        for i in range(T):
            pv_i = float(pv[0, i]) if pv.dim() == 2 else float(pv[0])
            pa_i = float(pa[0, i]) if pa.dim() == 2 else float(pa[0])
            tv_i = float(tv[0, i]) if tv.dim() == 2 else float(tv[0])
            ta_i = float(ta[0, i]) if ta.dim() == 2 else float(ta[0])
            print(f"t={i:02d}: valence {pv_i:+.3f}/{tv_i:+.3f} | arousal {pa_i:+.3f}/{ta_i:+.3f}")

    if "emotions" in preds:
        print("\n>>> Checking emotion logits...")
        logits = preds["emotions"]
        print(tstats(logits, "preds[emotions]"))
        pred_idx = logits.argmax(dim=1)
        acc = (pred_idx == targets["emotions"]).float().mean().item()
        print(f"batch top-1 acc: {acc:.3f}")

    print("\n✅ Step 2 debug completed successfully.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/ckvideo_out")
    ap.add_argument("--model_type", default="multitask", choices=["va_only", "emotions_only", "multitask"])
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--height", type=int, default=112)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--num_emotions", type=int, default=50)
    ap.add_argument("--efficient", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    main(args)

