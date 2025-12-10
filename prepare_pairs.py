# prepare_pairs.py
import json
import random
import itertools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_POLY_DIR = DATA_DIR / "raw" / "polyvore_outfits"
IMAGES_DIR = RAW_POLY_DIR / "images"  
PAIRS_CSV = DATA_DIR / "pairs.csv"

ITEM_META_PATH = RAW_POLY_DIR / "polyvore_item_metadata.json"

# For a quick prototype; set to None to use all outfits
MAX_OUTFITS = 500
NEG_RATIO = 1.0   
VAL_RATIO = 0.2   

random.seed(42)


def load_item_metadata():
    """Load item_id -> relative image path mapping."""
    print(f"Loading item metadata from {ITEM_META_PATH}")
    with open(ITEM_META_PATH, "r") as f:
        meta = json.load(f)

    id_to_img = {}
    for item_id, info in meta.items():
        img_path = info.get("image") or info.get("img") or ""
        if img_path.startswith("images/"):
            img_path = img_path[len("images/"):]
        if not img_path:
            img_path = f"{item_id}.jpg"
        id_to_img[item_id] = img_path
    print(f"Loaded metadata for {len(id_to_img)} items.")
    return id_to_img


def load_outfits_from_json(json_path: Path, id_to_img, max_outfits=None):
    """
    Each entry in train.json / test.json is an outfit.
    We map item_ids → image filenames using item metadata.
    """
    print(f"\nLoading outfits from {json_path}")
    with open(json_path, "r") as f:
        outfits_raw = json.load(f)

    outfits = []
    for outfit in outfits_raw:
        items = []

        for it in outfit.get("items", []):
            item_id = str(it.get("item_id"))
            if item_id not in id_to_img:
                continue
            img_rel = id_to_img[item_id]
            img_path = IMAGES_DIR / img_rel
            if img_path.exists():
                items.append(img_rel)

        if len(items) >= 2:
            outfits.append(items)

        if max_outfits is not None and len(outfits) >= max_outfits:
            break

    print(f"Kept {len(outfits)} outfits with >= 2 items.")
    return outfits


def generate_pairs(outfits, split_name: str, neg_ratio: float):
    """
    Given a list of outfits (each a list of image relpaths), build:
    - positive pairs (intra-outfit)
    - negative pairs (random across items)
    """
    all_items = list({img for outfit in outfits for img in outfit})

    pos_pairs = []
    for outfit in outfits:
        for a, b in itertools.combinations(outfit, 2):
            pos_pairs.append((a, b))
    print(f"{split_name}: {len(pos_pairs)} positive pairs.")

    pos_set = set(tuple(sorted(p)) for p in pos_pairs)

    num_neg = int(len(pos_pairs) * neg_ratio)
    neg_pairs = []
    attempts = 0
    max_attempts = num_neg * 10

    while len(neg_pairs) < num_neg and attempts < max_attempts:
        a, b = random.sample(all_items, 2)
        key = tuple(sorted((a, b)))
        if key not in pos_set:
            neg_pairs.append((a, b))
        attempts += 1

    print(f"{split_name}: {len(neg_pairs)} negative pairs.")

    rows = []
    for a, b in pos_pairs:
        rows.append({"img1": a, "img2": b, "label": 1, "split": split_name})
    for a, b in neg_pairs:
        rows.append({"img1": a, "img2": b, "label": 0, "split": split_name})
    return rows


def main():
    id_to_img = load_item_metadata()

    train_json = RAW_POLY_DIR / "disjoint" / "train.json"
    test_json  = RAW_POLY_DIR / "disjoint" / "test.json"

    train_outfits_all = load_outfits_from_json(
        train_json, id_to_img, max_outfits=MAX_OUTFITS
    )

    
    n_total = len(train_outfits_all)
    n_val = max(1, int(n_total * VAL_RATIO))
    val_outfits = train_outfits_all[:n_val]
    train_outfits = train_outfits_all[n_val:]
    print(f"\nSplit {n_total} train outfits into {len(train_outfits)} train and {len(val_outfits)} val.")

    test_outfits = load_outfits_from_json(
        test_json, id_to_img, max_outfits=MAX_OUTFITS
    )

    rows = []
    rows += generate_pairs(train_outfits, "train", NEG_RATIO)
    rows += generate_pairs(val_outfits,   "val",   NEG_RATIO)
    rows += generate_pairs(test_outfits,  "test",  NEG_RATIO)

    df = pd.DataFrame(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PAIRS_CSV, index=False)
    print(f"\n✅ Saved {len(df)} pairs to {PAIRS_CSV}")


if __name__ == "__main__":
    main()
