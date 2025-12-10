import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from core import (
    IMAGES_DIR,
    load_trained_models_for_inference,
    predict_pair,
)

DATA_DIR = Path("data")
PAIRS_PATH = DATA_DIR / "pairs.csv"

random.seed(42)
np.random.seed(42)


def load_pairs(split: str, max_pairs: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(PAIRS_PATH)
    df = df[df["split"] == split].reset_index(drop=True)
    if max_pairs is not None and max_pairs < len(df):
        df = df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
    return df

def compute_color_hist(img_path: Path, cache: dict, bins: int = 8) -> np.ndarray:
    key = str(img_path)
    if key in cache:
        return cache[key]

    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))

    hsv = img.convert("HSV")
    arr = np.array(hsv)

    h = arr[:, :, 0].flatten()
    s = arr[:, :, 1].flatten()
    v = arr[:, :, 2].flatten()

    hist_h, _ = np.histogram(h, bins=bins, range=(0, 256), density=False)
    hist_s, _ = np.histogram(s, bins=bins, range=(0, 256), density=False)
    hist_v, _ = np.histogram(v, bins=bins, range=(0, 256), density=False)

    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    norm = np.linalg.norm(hist) + 1e-8
    hist = hist / norm

    cache[key] = hist
    return hist


def compute_color_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    cache: dict[str, np.ndarray] = {}

    for _, row in df.iterrows():
        img1_path = IMAGES_DIR / row["img1"]
        img2_path = IMAGES_DIR / row["img2"]

        h1 = compute_color_hist(img1_path, cache)
        h2 = compute_color_hist(img2_path, cache)

        sim = float(np.dot(h1, h2)) 
        scores.append(sim)
        labels.append(int(row["label"]))

    return np.array(scores), np.array(labels)

def compute_clip_scores(
    df: pd.DataFrame,
    clip_model,
    preprocess,
    classifier,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_sims = []
    probs = []
    labels = []

    clip_model.eval()
    classifier.eval()

    with torch.no_grad():
        for _, row in df.iterrows():
            img1_path = IMAGES_DIR / row["img1"]
            img2_path = IMAGES_DIR / row["img2"]

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            cos_sim, prob = predict_pair(
                clip_model, preprocess, classifier, img1, img2
            )

            cos_sims.append(float(cos_sim))
            probs.append(float(prob))
            labels.append(int(row["label"]))

    return np.array(cos_sims), np.array(probs), np.array(labels)

def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
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


def main(max_pairs: int | None = None, hard_neg_percentile: float = 0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess, classifier = load_trained_models_for_inference()
    clip_model.to(device)
    classifier.to(device)

    print("\n==============================")
    print(" HARD NEGATIVE EVAL (VAL→TEST)")
    print("==============================")

    df_val = load_pairs(split="val", max_pairs=max_pairs)
    print(f"Loaded {len(df_val)} validation pairs.")

    color_val, y_val = compute_color_scores(df_val)
    cos_val, prob_val, y_val2 = compute_clip_scores(df_val, clip_model, preprocess, classifier)
    assert np.array_equal(y_val, y_val2)

    t_color = best_threshold(color_val, y_val)
    t_clip = best_threshold(cos_val, y_val)
    t_mlp = best_threshold(prob_val, y_val)

    print("\nChosen thresholds from VAL:")
    print(f"  Color hist threshold: {t_color:.4f}")
    print(f"  CLIP cosine thresh  : {t_clip:.4f}")
    print(f"  MLP prob thresh     : {t_mlp:.4f}")

    df_test = load_pairs(split="test", max_pairs=max_pairs)
    print(f"\nLoaded {len(df_test)} test pairs.")

    color_test, y_test = compute_color_scores(df_test)
    cos_test, prob_test, y_test2 = compute_clip_scores(df_test, clip_model, preprocess, classifier)
    assert np.array_equal(y_test, y_test2)

    neg_mask = (y_test == 0)
    pos_mask = (y_test == 1)

    color_neg = color_test[neg_mask]

    thresh_hard = np.quantile(color_neg, hard_neg_percentile)
    hard_neg_indices = np.where(neg_mask & (color_test >= thresh_hard))[0]

    num_hard_neg = len(hard_neg_indices)
    print(f"\nFound {num_hard_neg} hard negatives (top {int(hard_neg_percentile*100)}% by color similarity among negatives).")

    if num_hard_neg == 0:
        print("No hard negatives found with this percentile; try lowering hard_neg_percentile.")
        return

    pos_indices = np.where(pos_mask)[0]
    if len(pos_indices) < num_hard_neg:
        num_hard_neg = len(pos_indices)
        hard_neg_indices = hard_neg_indices[:num_hard_neg]
        print(f"Clipped hard negatives to {num_hard_neg} to match available positives.")

    hard_pos_indices = np.random.choice(pos_indices, size=num_hard_neg, replace=False)

    subset_indices = np.concatenate([hard_neg_indices, hard_pos_indices])
    subset_labels = y_test[subset_indices]

    print(f"Hard subset size: {len(subset_indices)} ({num_hard_neg} negatives, {num_hard_neg} positives)")

    color_hard = color_test[subset_indices]
    cos_hard = cos_test[subset_indices]
    prob_hard = prob_test[subset_indices]

    color_stats = eval_model(color_hard, subset_labels, t_color, name="Color hist baseline (HARD TEST)")
    clip_stats = eval_model(cos_hard, subset_labels, t_clip, name="CLIP cosine baseline (HARD TEST)")
    mlp_stats = eval_model(prob_hard, subset_labels, t_mlp, name="CLIP + MLP (HARD TEST)")

    out = {
        "hard_neg_percentile": hard_neg_percentile,
        "val_num_pairs": int(len(df_val)),
        "test_num_pairs": int(len(df_test)),
        "hard_subset": {
            "num_pairs": int(len(subset_indices)),
            "num_neg": int(num_hard_neg),
            "num_pos": int(num_hard_neg),
        },
        "color_hist": color_stats,
        "clip_cosine": clip_stats,
        "mlp": mlp_stats,
    }

    import json
    with open("eval_hardneg_results.json", "w") as f:
        json.dump(out, f, indent=4)
    print("\nSaved hard-negative evaluation results to eval_hardneg_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="optionally subsample val+test for speed",
    )
    parser.add_argument(
        "--hard_neg_percentile",
        type=float,
        default=0.8,
        help="percentile (0-1) of color similarity among negative pairs to define 'hard' negatives",
    )
    args = parser.parse_args()

    main(max_pairs=args.max_pairs, hard_neg_percentile=args.hard_neg_percentile)
