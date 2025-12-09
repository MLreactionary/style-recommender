import argparse
from pathlib import Path
import json

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

# Paths
DATA_DIR = Path("data")
PAIRS_PATH = DATA_DIR / "pairs.csv"
RAW_POLY_DIR = DATA_DIR / "raw" / "polyvore_outfits"
ITEM_META_PATH = RAW_POLY_DIR / "polyvore_item_metadata.json"


# ---------- Utility: load pairs ----------

def load_pairs(split: str, max_pairs: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(PAIRS_PATH)
    df = df[df["split"] == split].reset_index(drop=True)
    if max_pairs is not None and max_pairs < len(df):
        df = df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
    return df


# ---------- Utility: item metadata → categories ----------

def load_img_to_category() -> dict:
    """
    Build a mapping: image_rel_path -> semantic_category (string),
    using polyvore_item_metadata.json.

    We try several likely keys: 'semantic_category', 'category', 'categoryid'.
    """
    print(f"Loading item metadata from {ITEM_META_PATH}")
    with open(ITEM_META_PATH, "r") as f:
        meta = json.load(f)

    img_to_cat: dict[str, str] = {}

    for item_id, info in meta.items():
        # image path
        img_path = info.get("image") or info.get("img") or ""
        if img_path.startswith("images/"):
            img_path = img_path[len("images/"):]
        if not img_path:
            img_path = f"{item_id}.jpg"

        # category
        cat = (
            info.get("semantic_category")
            or info.get("category")
            or info.get("categoryid")
            or "unknown"
        )
        cat_str = str(cat).lower()

        img_to_cat[img_path] = cat_str

    print(f"Built img_to_cat mapping for {len(img_to_cat)} images.")
    return img_to_cat


def annotate_categories(df: pd.DataFrame, img_to_cat: dict) -> pd.DataFrame:
    def get_cat(relpath: str) -> str:
        # relpath in pairs.csv is something like "disjoint/train/...jpg" or just "12345.jpg"
        # We only stored the tail when building outfits (e.g., "12345.jpg"),
        # but to be safe, strip directory portions.
        rel = Path(relpath).name
        return img_to_cat.get(rel, "unknown")

    df = df.copy()
    df["cat1"] = df["img1"].apply(get_cat)
    df["cat2"] = df["img2"].apply(get_cat)
    return df


def highlevel_cat(cat: str) -> str:
    """
    Map raw category strings to coarse types: 'top', 'bottom', 'shoe', 'other'.
    Very heuristic / approximate, but good enough for slices.
    """
    c = cat.lower()

    top_keywords = ["top", "blouse", "shirt", "tee", "t-shirt", "sweater", "hoodie", "jacket", "coat", "cardigan"]
    bottom_keywords = ["pant", "pants", "jean", "short", "skirt", "trouser", "legging"]
    shoe_keywords = ["shoe", "sandal", "boot", "heel", "sneaker", "loafer", "flat"]

    if any(k in c for k in top_keywords):
        return "top"
    if any(k in c for k in bottom_keywords):
        return "bottom"
    if any(k in c for k in shoe_keywords):
        return "shoe"
    return "other"


# ---------- Low-level color histogram (same as earlier) ----------

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


def compute_color_scores(df: pd.DataFrame) -> np.ndarray:
    scores = []
    cache: dict[str, np.ndarray] = {}

    for _, row in df.iterrows():
        img1_path = IMAGES_DIR / row["img1"]
        img2_path = IMAGES_DIR / row["img2"]

        h1 = compute_color_hist(img1_path, cache)
        h2 = compute_color_hist(img2_path, cache)
        sim = float(np.dot(h1, h2))
        scores.append(sim)

    return np.array(scores)


# ---------- CLIP + MLP scores ----------

def compute_clip_scores(
    df: pd.DataFrame,
    clip_model,
    preprocess,
    classifier,
) -> tuple[np.ndarray, np.ndarray]:
    cos_sims = []
    probs = []

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

    return np.array(cos_sims), np.array(probs)


# ---------- Threshold + eval ----------

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


def eval_model(scores: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")
    return acc, f1, auc


# ---------- Main: slice-based evaluation ----------

def main(max_pairs: int | None = None, min_slice_size: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess, classifier = load_trained_models_for_inference()
    clip_model.to(device)
    classifier.to(device)

    print("\n==============================")
    print(" SLICE-BASED EVALUATION (VAL→TEST)")
    print("==============================")

    # ----- 1) Load val and tune thresholds globally -----
    df_val = load_pairs("val", max_pairs=max_pairs)
    print(f"Loaded {len(df_val)} validation pairs.")

    labels_val = df_val["label"].values.astype(int)

    color_val = compute_color_scores(df_val)
    cos_val, prob_val = compute_clip_scores(df_val, clip_model, preprocess, classifier)

    t_color = best_threshold(color_val, labels_val)
    t_clip = best_threshold(cos_val, labels_val)
    t_mlp = best_threshold(prob_val, labels_val)

    print("\nGlobal thresholds (from VAL):")
    print(f"  Color hist: {t_color:.4f}")
    print(f"  CLIP cos  : {t_clip:.4f}")
    print(f"  MLP prob  : {t_mlp:.4f}")

    # ----- 2) Load test + annotate categories -----
    df_test = load_pairs("test", max_pairs=max_pairs)
    print(f"\nLoaded {len(df_test)} test pairs.")

    img_to_cat = load_img_to_category()
    df_test = annotate_categories(df_test, img_to_cat)

    # High-level category mapping
    df_test["cat1_high"] = df_test["cat1"].apply(highlevel_cat)
    df_test["cat2_high"] = df_test["cat2"].apply(highlevel_cat)

    labels_test = df_test["label"].values.astype(int)
    color_test = compute_color_scores(df_test)
    cos_test, prob_test = compute_clip_scores(df_test, clip_model, preprocess, classifier)

    # also compute median color sim for color-similar vs dissimilar slices
    median_color = float(np.median(color_test))
    print(f"\nMedian color similarity on TEST: {median_color:.4f}")

    # ----- 3) Define slices -----
    masks = {}

    # All test
    masks["all"] = np.ones(len(df_test), dtype=bool)

    # Same raw category (ignoring 'unknown')
    same_cat = (df_test["cat1"] == df_test["cat2"]) & (df_test["cat1"] != "unknown")
    masks["same_raw_category"] = same_cat.values

    # Different raw category (both known)
    diff_cat = (df_test["cat1"] != df_test["cat2"]) & (df_test["cat1"] != "unknown") & (df_test["cat2"] != "unknown")
    masks["different_raw_category"] = diff_cat.values

    # Top-bottom combos (one top, one bottom)
    top_bottom = (
        ((df_test["cat1_high"] == "top") & (df_test["cat2_high"] == "bottom"))
        | ((df_test["cat1_high"] == "bottom") & (df_test["cat2_high"] == "top"))
    )
    masks["top_bottom_pairs"] = top_bottom.values

    # Both shoes
    both_shoes = (df_test["cat1_high"] == "shoe") & (df_test["cat2_high"] == "shoe")
    masks["shoe_shoe_pairs"] = both_shoes.values

    # Color-similar vs color-dissimilar
    masks["high_color_similarity"] = color_test >= median_color
    masks["low_color_similarity"] = color_test < median_color

    # ----- 4) Evaluate on each slice -----
    results = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n < min_slice_size:
            print(f"\nSlice '{name}' has only {n} pairs (< {min_slice_size}); skipping.")
            continue

        y = labels_test[mask]
        col = color_test[mask]
        cos = cos_test[mask]
        prob = prob_test[mask]

        acc_c, f1_c, auc_c = eval_model(col, y, t_color)
        acc_clip, f1_clip, auc_clip = eval_model(cos, y, t_clip)
        acc_mlp, f1_mlp, auc_mlp = eval_model(prob, y, t_mlp)

        print(f"\n=== Slice: {name} (n={n}) ===")
        print(f"  Color hist   - Acc: {acc_c:.3f}, F1: {f1_c:.3f}, AUC: {auc_c:.3f}")
        print(f"  CLIP cosine  - Acc: {acc_clip:.3f}, F1: {f1_clip:.3f}, AUC: {auc_clip:.3f}")
        print(f"  CLIP + MLP   - Acc: {acc_mlp:.3f}, F1: {f1_mlp:.3f}, AUC: {auc_mlp:.3f}")

        results[name] = {
            "n_pairs": n,
            "color_hist": {"acc": acc_c, "f1": f1_c, "auc": auc_c},
            "clip_cosine": {"acc": acc_clip, "f1": f1_clip, "auc": auc_clip},
            "mlp": {"acc": acc_mlp, "f1": f1_mlp, "auc": auc_mlp},
        }

    # ----- 5) Save slice results -----
    out = {
        "thresholds": {
            "color_hist": float(t_color),
            "clip": float(t_clip),
            "mlp": float(t_mlp),
        },
        "slices": results,
    }

    with open("eval_slices_results.json", "w") as f:
        json.dump(out, f, indent=4)
    print("\nSaved slice-based evaluation to eval_slices_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=None,
        help="optionally subsample val+test for speed",
    )
    parser.add_argument(
        "--min_slice_size",
        type=int,
        default=100,
        help="minimum slice size to report results",
    )
    args = parser.parse_args()

    main(max_pairs=args.max_pairs, min_slice_size=args.min_slice_size)
