import os, sys, json
import pandas as pd
import numpy as np

def summarize_split(csv_path, name):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    out = {"name": name, "rows": int(len(df)), "cols": int(len(df.columns))}
    for req in ("id","type","arousal","valence"):
        out[f"has_{req}"] = (req in df.columns)

    # VA stats
    if "valence" in df.columns and "arousal" in df.columns:
        v = pd.to_numeric(df["valence"], errors="coerce")
        a = pd.to_numeric(df["arousal"], errors="coerce")
        out.update({
            "valence_min": float(v.min()),
            "valence_max": float(v.max()),
            "valence_mean": float(v.mean()),
            "arousal_min": float(a.min()),
            "arousal_max": float(a.max()),
            "arousal_mean": float(a.mean()),
            "va_nan_frac": float(((v.isna()) | (a.isna())).mean()),
        })

    # base_id & frame coverage
    if "id" in df.columns:
        base = df["id"].astype(str).str.split("_").str[0]
        out["unique_videos"] = int(base.nunique())
        out["avg_rows_per_video"] = float(len(df) / max(1, base.nunique()))
        df["base_id"] = base
        df["frame"] = df["id"].astype(str).str.split("_").str[1].str.zfill(3)
    return out, df

def load_emotion_meta(meta_csv):
    if not os.path.isfile(meta_csv):
        return None, []
    meta = pd.read_csv(meta_csv, low_memory=False)
    fn = "Filename" if "Filename" in meta.columns else ("filename" if "filename" in meta.columns else None)
    if fn is None:
        return None, []
    meta = meta.copy()
    meta["base_id"] = meta[fn].astype(str).str.replace(".mp4","", regex=False)
    drop = {'filename','Filename','valence','Valence','arousal','Arousal','base_id'}
    emo_cols = [c for c in meta.columns if c not in drop]
    return meta[["base_id"]+emo_cols], emo_cols

def topk_emotions(df, k=50):
    skip = {'filename','valence','arousal','id','type','base_id','frame'}
    emo_cols = [c for c in df.columns if c not in skip]
    if not emo_cols:
        return []
    emo_df = df[emo_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts = emo_df.sum().sort_values(ascending=False)
    return list(counts.head(k).index)

def attach_meta(df, meta):
    if meta is None:
        return df
    if "base_id" not in df.columns:
        df = df.copy()
        df["base_id"] = df["id"].astype(str).str.split("_").str[0]
    return df.merge(meta, on="base_id", how="left")

if __name__ == "__main__":
    # Usage:
    #   python debug_step1_data.py [data_root]
    # Defaults to data/ckvideo_out
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/ckvideo_out"
    train_csv = os.path.join(data_root, "splits", "train.csv")
    val_csv   = os.path.join(data_root, "splits", "val.csv")
    meta_csv  = os.path.join(data_root, "metadata", "CowenKeltnerEmotionalVideos.csv")

    train_info, train_df = summarize_split(train_csv, "train")
    val_info,   val_df   = summarize_split(val_csv,   "val")

    print("=== SPLIT SUMMARY ===")
    print(json.dumps(train_info, indent=2))
    print(json.dumps(val_info, indent=2))

    # Load metadata and compute Top-K emotion columns for mapping check
    meta, meta_cols = load_emotion_meta(meta_csv)
    train_joined = attach_meta(train_df, meta)
    val_joined   = attach_meta(val_df,   meta)

    top_train = topk_emotions(train_joined, k=50)
    top_val   = topk_emotions(val_joined,   k=50)

    print("\n=== EMOTION MAPPING CHECK ===")
    print("Top-K (train) head:", top_train[:10])
    print("Top-K (val)   head:", top_val[:10])
    mapping_equal = (top_train == top_val)
    print("MAPPING MATCH:", bool(mapping_equal))
    if not mapping_equal:
        print("First mismatches (up to 20):")
        for i, (a,b) in enumerate(zip(top_train, top_val)):
            if a != b:
                print(f"  @idx {i}: train={a}  val={b}")
                if i >= 19:
                    break

    # Heuristics for VA scaling
    def scale_hint(info, name):
        vmin, vmax = info.get("valence_min"), info.get("valence_max")
        amin, amax = info.get("arousal_min"), info.get("arousal_max")
        if vmin is None or vmax is None or amin is None or amax is None:
            return
        print(f"\n=== VA SCALE HINT ({name}) ===")
        print(f"valence range ~ [{vmin:.3f}, {vmax:.3f}]  |  arousal range ~ [{amin:.3f}, {amax:.3f}]")
        if vmax > 2.0 or amax > 2.0:
            print("-> Looks like raw scale (e.g., 0..10). If your model outputs are bounded "
                  "to [-1,1] / [0,1], you must normalize targets accordingly.")
    scale_hint(train_info, "train")
    scale_hint(val_info,   "val")

    # Frame coverage quick check
    print("\n=== FRAME COVERAGE (train) ===")
    if "frame" in train_df.columns:
        per_video = train_df.groupby("base_id")["frame"].nunique()
        print(f"videos: {per_video.size}, frames-per-video: min={int(per_video.min())}, "
              f"max={int(per_video.max())}, mean={float(per_video.mean()):.2f}")

