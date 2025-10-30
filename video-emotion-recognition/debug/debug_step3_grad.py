# debug/debug_step3_grad.py
import os, sys, json, random, argparse, time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# If running without -m, uncomment the shim:
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

from data_utils.dataset import create_ck_data_loaders
from models.video_rcnn import get_model
from models.losses import get_loss_function

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def grad_norm(module):
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += g.norm().item()
            count += 1
    return (total / max(1, count)) if count > 0 else 0.0

def tstats(t, name):
    t_cpu = t.detach().float().cpu()
    return f"{name} shape={tuple(t.shape)} min={t_cpu.min():.4f} max={t_cpu.max():.4f} mean={t_cpu.mean():.4f} std={t_cpu.std():.4f}"

def main(args):
    print(">>> Setting seed")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device}")

    print("\n>>> Building data loaders")
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
    print(json.dumps(info, indent=2)[:2000])

    print("\n>>> Fetching one batch")
    videos, targets = next(iter(train_loader))
    print(f"videos (B,T,C,H,W) = {tuple(videos.shape)}")
    for k,v in targets.items():
        print(f"targets[{k}] = {tuple(v.shape)}")

    print("\n>>> Moving to device & permute to (B,C,T,H,W)")
    videos = videos.to(device).permute(0,2,1,3,4).contiguous()
    targets = {k: v.to(device) for k,v in targets.items()}
    print("videos.device:", videos.device)

    print("\n>>> Building model & loss")
    model = get_model(
        model_type=args.model_type,
        num_frames=args.num_frames,
        num_emotions=info.get("num_emotions", 50),
        efficient=args.efficient,
    ).to(device)
    model.train()
    criterion = get_loss_function(model_type=args.model_type, adaptive_weights=False)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Forward 1
    print("\n>>> Forward pass #1")
    outputs = model(videos)
    for k,v in outputs.items():
        print(tstats(v, f"outputs[{k}]"))

    # Loss 1
    print("\n>>> Compute loss #1")
    loss_dict = criterion(outputs, targets)
    total1 = loss_dict["total_loss"]
    print(f"total_loss #1: {float(total1):.6f}")
    for k,v in loss_dict.items():
        if k != "total_loss" and torch.is_tensor(v):
            print(f"  {k}: {float(v):.6f}")

    print("\n>>> Backward #1")
    optimizer.zero_grad()
    total1.backward()

    # Gradient norms
    print("\n=== Gradient norms (mean over params with grad) ===")
    # Backbone bits (best effort — names exist in your model)
    try:
        print("conv1:", grad_norm(model.backbone.conv1))
        print("rconv1:", grad_norm(model.backbone.rconv1))
        print("rconv2:", grad_norm(model.backbone.rconv2))
        print("rconv3:", grad_norm(model.backbone.rconv3))
        print("rconv4:", grad_norm(model.backbone.rconv4))
    except Exception as e:
        print("[note] couldn't fetch some backbone grads:", e)

    # Heads
    if "val_h" in model.__dict__ or hasattr(model, "val_h"):
        print("val_h:", grad_norm(model.val_h))
    if "aro_h" in model.__dict__ or hasattr(model, "aro_h"):
        print("aro_h:", grad_norm(model.aro_h))
    if hasattr(model, "shared_features"):
        print("shared_features:", grad_norm(model.shared_features))
    if hasattr(model, "emotion_head"):
        print("emotion_head:", grad_norm(model.emotion_head))

    print("\n>>> Optimizer step")
    optimizer.step()

    # Forward 2 (after one step)
    print("\n>>> Forward pass #2")
    outputs2 = model(videos)
    loss_dict2 = criterion(outputs2, targets)
    total2 = loss_dict2["total_loss"]
    print(f"total_loss #2: {float(total2):.6f}")
    for k,v in loss_dict2.items():
        if k != "total_loss" and torch.is_tensor(v):
            print(f"  {k}: {float(v):.6f}")

    # Optional: show a tiny alignment head on sample 0
    if "valence" in outputs2 and "arousal" in outputs2:
        pv = outputs2["valence"].detach().cpu()
        pa = outputs2["arousal"].detach().cpu()
        tv = targets["valence"].detach().cpu()
        ta = targets["arousal"].detach().cpu()
        T = min(pv.shape[-1], tv.shape[-1])
        print("\nFirst sample (after 1 step) head:")
        for i in range(min(T, 10)):
            print(f"  t={i:02d}: v {pv[0,i]:+.3f}/{tv[0,i]:+.3f} | a {pa[0,i]:+.3f}/{ta[0,i]:+.3f}")

    print("\n✅ Step 3 done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/ckvideo_out")
    ap.add_argument("--model_type", default="multitask", choices=["va_only","emotions_only","multitask"])
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--height", type=int, default=112)
    ap.add_argument("--width", type=int, default=112)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--num_emotions", type=int, default=50)
    ap.add_argument("--efficient", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    main(args)

