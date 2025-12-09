import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from core import (
    load_trained_models_for_inference,
    predict_pair,
    IMAGES_DIR,
)


DATA_DIR = Path("data")
PAIRS_PATH = DATA_DIR / "pairs.csv"


def load_pairs(split: str = "test", max_pairs: int | None = None) -> pd.DataFrame:
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


def compute_scores(
    df: pd.DataFrame,
    clip_model,
    preprocess,
    classifier,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pair row, compute:
        - cos_sim: CLIP cosine similarity baseline
        - prob:    MLP compatibility probability
        - y_true:  ground-truth label (0/1)

    Returns:
        cos_sims, probs, labels  (each as 1D numpy arrays)
    """
    cos_sims = []
    probs = []
    labels = []

    clip_model.eval()
    classifier.eval()

    device = next(classifier.parameters()).device

    with torch.no_grad():
        for _, row in df.iterrows():
            img1_path = IMAGES_DIR / row["img1"]
            img2_path = IMAGES_DIR / row["img2"]

            # Load as PIL images
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # Reuse your existing helper: returns (cos_sim, prob)
            cos_sim, prob = predict_pair(
            clip_model, preprocess, classifier, img1, img2
            )


            cos_sims.append(float(cos_sim))
            probs.append(float(prob))
            labels.append(int(row["label"]))

    return np.array(cos_sims), np.array(probs), np.array(labels)


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the threshold on `scores` (e.g., cosine similarity or probability)
    that maximizes F1 on the given labels.
    """
    best_t = 0.5
    best_f1 = -1.0

    # Try thresholds between min and max
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


def main(max_pairs: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained models
    clip_model, preprocess, classifier = load_trained_models_for_inference()
    clip_model.to(device)
    classifier.to(device)
    clip_model.eval()
    classifier.eval()

    # ---------- 1) TUNE THRESHOLDS ON VALID ----------
    print("\n======================")
    print(" TUNING ON VALIDATION ")
    print("======================")

    df_val = load_pairs(split="val", max_pairs=max_pairs)
    print(f"Loaded {len(df_val)} validation pairs.")

    cos_val, prob_val, y_val = compute_scores(df_val, clip_model, preprocess, classifier)

    t_clip = best_threshold(cos_val, y_val)
    t_prob = best_threshold(prob_val, y_val)

    print(f"\nChosen thresholds based on VALIDATION set:")
    print(f"  CLIP baseline threshold: {t_clip:.4f}")
    print(f"  MLP  baseline threshold: {t_prob:.4f}")

    # ---------- 2) EVALUATE ON TEST USING THOSE THRESHOLDS ----------
    print("\n=================")
    print(" TEST EVALUATION ")
    print("=================")

    df_test = load_pairs(split="test", max_pairs=max_pairs)
    print(f"Loaded {len(df_test)} test pairs.")

    cos_test, prob_test, y_test = compute_scores(df_test, clip_model, preprocess, classifier)

    eval_model(cos_test, y_test, t_clip, name="CLIP cosine-only baseline (TEST)")
    eval_model(prob_test, y_test, t_prob, name="CLIP + MLP (compatibility head) (TEST)")


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

