# eval_lowlevel_baselines.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from core import IMAGES_DIR  # just need this for image paths


DATA_DIR = Path("data")
PAIRS_PATH = DATA_DIR / "pairs.csv"


def load_pairs(split: str, max_pairs: int | None = None) -> pd.DataFrame:
    """
    Load pair rows for a given split from pairs.csv.

    Assumes pairs.csv has at least columns:
        img1, img2, label, split
    """
    df = pd.read_csv(PAIRS_PATH)
    df = df[df["split"] == split].reset_index(drop=True)
    if max_pairs is not None and max_pairs < len(df):
        df = df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
    return df


# ---------- Low-level feature: color histogram ----------

def compute_color_hist(img_path: Path, cache: dict, bins: int = 8) -> np.ndarray:
    """
    Compute a simple HSV color histogram for an image and L2-normalize it.

    To speed things up, we cache histograms in the 'cache' dict keyed by str(img_path).
    """
    key = str(img_path)
    if key in cache:
        return cache[key]

    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))  # downsample to speed up

    # Convert to HSV and compute histogram on each channel
    hsv = img.convert("HSV")
    arr = np.array(hsv)

    h = arr[:, :, 0].flatten()
    s = arr[:, :, 1].flatten()
    v = arr[:, :, 2].flatten()

    hist_h, _ = np.histogram(h, bins=bins, range=(0, 256), density=False)
    hist_s, _ = np.histogram(s, bins=bins, range=(0, 256), density=False)
    hist_v, _ = np.histogram(v, bins=bins, range=(0, 256), density=False)

    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

    # L2-normalize
    norm = np.linalg.norm(hist) + 1e-8
    hist = hist / norm

    cache[key] = hist
    return hist


def compute_color_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    For each pair row, compute a color-histogram similarity score and return:
        scores, labels
    where scores is 1D np.ndarray of similarities,
    labels is 1D np.ndarray of 0/1 compatibility labels.
    """
    scores = []
    labels = []

    cache: dict[str, np.ndarray] = {}

    for _, row in df.iterrows():
        img1_path = IMAGES_DIR / row["img1"]
        img2_path = IMAGES_DIR / row["img2"]

        h1 = compute_color_hist(img1_path, cache)
        h2 = compute_color_hist(img2_path, cache)

        # cosine similarity between normalized histograms
        sim = float(np.dot(h1, h2))

        scores.append(sim)
        labels.append(int(row["label"]))

    return np.array(scores), np.array(labels)


# ---------- Generic helpers for threshold + metrics ----------

def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the threshold on `scores` that maximizes F1 on the given labels.
    """
    best_t = 0.5
    best_f1 = -1.0

    ts = np.linspace(scores.min(), scores.max(), num=200)
    for t in ts:
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def eval_model(scores: np.ndarray, labels: np.ndarray, threshold: float, name: str):
    """
    Print Accuracy, F1, and AUC for a given set of scores + labels and threshold.
    Returns a dict of stats for saving.
    """
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    print(f"\n=== {name} ===")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "f1": float(f1),
        "auc": float(auc),
    }


def main(max_pairs: int | None = None):
    print("\n==============================")
    print(" COLOR HISTOGRAM BASELINE EVAL")
    print("  (tuned on val, tested on test)")
    print("==============================")

    # ---------- 1) TUNE THRESHOLD ON VAL ----------
    df_val = load_pairs(split="val", max_pairs=max_pairs)
    print(f"Loaded {len(df_val)} validation pairs.")

    scores_val, labels_val = compute_color_scores(df_val)
    t_color = best_threshold(scores_val, labels_val)

    print(f"\nChosen color-hist threshold (VAL): {t_color:.4f}")

    # ---------- 2) EVALUATE ON TEST USING THAT THRESHOLD ----------
    df_test = load_pairs(split="test", max_pairs=max_pairs)
    print(f"\nLoaded {len(df_test)} test pairs.")

    scores_test, labels_test = compute_color_scores(df_test)

    color_stats = eval_model(
        scores_test, labels_test, t_color, name="Color histogram baseline (TEST)"
    )

    # ---------- 3) Save results to JSON for later use ----------
    out = {
        "thresholds": {
            "color_hist": float(t_color),
        },
        "validation": {
            "num_val_pairs": int(len(df_val)),
        },
        "test": {
            "num_test_pairs": int(len(df_test)),
            "color_hist": color_stats,
        },
    }

    import json
    with open("eval_lowlevel_results.json", "w") as f:
        json.dump(out, f, indent=4)
    print("\nSaved low-level baseline results to eval_lowlevel_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="optionally subsample pairs for quick runs (applied to both val and test)",
    )
    args = parser.parse_args()

    main(max_pairs=args.max_pairs)
