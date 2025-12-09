# core.py
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import open_clip
import requests
from sklearn.cluster import KMeans

# ---------- Paths & config ----------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Raw Polyvore dataset root (your Kaggle layout)
RAW_POLY_DIR = DATA_DIR / "raw" / "polyvore_outfits"
IMAGES_DIR = RAW_POLY_DIR / "images"   # all item images live here
PAIRS_CSV = DATA_DIR / "pairs.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "compat_mlp.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- CLIP via open_clip ----------

def load_clip_model():
    """
    Load CLIP image encoder + preprocessing using open_clip, not transformers.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval()
    model.to(DEVICE)

    # freeze CLIP
    for p in model.parameters():
        p.requires_grad = False

    return model, preprocess


# ---------- Dataset for training ----------

class PairDataset(Dataset):
    """
    Reads (img1, img2, label, split) from pairs.csv and returns preprocessed tensors.
    CSV columns:
        img1, img2, label, split
    where img1/img2 are relative paths under IMAGES_DIR.
    """
    def __init__(self, split: str, preprocess, csv_path: Path = PAIRS_CSV):
        assert split in {"train", "val", "test"}
        self.preprocess = preprocess
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        img_path = IMAGES_DIR / rel_path
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = self._load_image(row["img1"])
        img2 = self._load_image(row["img2"])
        label = float(row["label"])

        # open_clip preprocess returns a (3,H,W) tensor
        pv1 = self.preprocess(img1)
        pv2 = self.preprocess(img2)

        return {
            "pixel_values1": pv1,
            "pixel_values2": pv2,
            "label": label,
        }


# ---------- Compatibility MLP ----------

class CompatibilityMLP(nn.Module):
    """
    Input: [emb1, emb2, |emb1-emb2|, emb1*emb2]  → logit.
    embed_dim = model.visual.output_dim (e.g., 512 for ViT-B-32).
    """
    def __init__(self, embed_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        input_dim = 4 * embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, emb1, emb2):
        diff = torch.abs(emb1 - emb2)
        prod = emb1 * emb2
        x = torch.cat([emb1, emb2, diff, prod], dim=-1)
        logit = self.net(x)
        return logit.squeeze(-1)  # (B,)


# ---------- Shared helpers ----------

def compute_batch_embeddings(clip_model, batch):
    """
    Given a batch from PairDataset, compute normalized image embeddings.
    """
    pv1 = batch["pixel_values1"].to(DEVICE)  # (B,3,H,W)
    pv2 = batch["pixel_values2"].to(DEVICE)

    with torch.no_grad():
        emb1 = clip_model.encode_image(pv1)
        emb2 = clip_model.encode_image(pv2)

    emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
    emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
    return emb1, emb2


def encode_single_image(clip_model, preprocess, img: Image.Image) -> torch.Tensor:
    """
    Encode a single PIL image into a normalized CLIP embedding.
    """
    pixel_values = preprocess(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

    with torch.no_grad():
        emb = clip_model.encode_image(pixel_values)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb  # (1, D)


def load_trained_models_for_inference():
    """
    Load frozen CLIP (open_clip) + trained CompatibilityMLP classifier from disk.
    Used by the Streamlit app.
    """
    clip_model, preprocess = load_clip_model()

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    embed_dim = checkpoint.get("embed_dim", clip_model.visual.output_dim)

    classifier = CompatibilityMLP(embed_dim=embed_dim)
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.to(DEVICE)
    classifier.eval()

    return clip_model, preprocess, classifier


def predict_pair(
    clip_model, preprocess, classifier, img1: Image.Image, img2: Image.Image
) -> Tuple[float, float]:
    """
    Given 2 PIL images, return:
        cosine_similarity (float),
        compatibility_probability (0–1).
    """
    emb1 = encode_single_image(clip_model, preprocess, img1)
    emb2 = encode_single_image(clip_model, preprocess, img2)

    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()

    with torch.no_grad():
        logit = classifier(emb1, emb2)
        prob = torch.sigmoid(logit).item()

    return cos_sim, prob


# ---------- New: dataset utilities for recommendation ----------

def load_pairs_df() -> pd.DataFrame:
    """Load pairs.csv once."""
    return pd.read_csv(PAIRS_CSV)


def get_test_image_pool() -> List[str]:
    """
    Return a list of unique image filenames that appear in the test split.
    """
    df = load_pairs_df()
    df_test = df[df["split"] == "test"]
    imgs = set(df_test["img1"]).union(set(df_test["img2"]))
    imgs = [img for img in imgs if isinstance(img, str)]
    return imgs


def sample_test_images(n: int) -> List[str]:
    """
    Sample up to n image filenames from the test image pool.
    """
    imgs = get_test_image_pool()
    if not imgs:
        return []
    n = min(n, len(imgs))
    rng = np.random.default_rng()
    return list(rng.choice(imgs, size=n, replace=False))


# def rank_candidates_for_anchor(
#     clip_model,
#     preprocess,
#     classifier,
#     anchor_img: Image.Image,
#     candidate_paths: List[str],
#     top_k: int = 8,
# ) -> List[Dict]:
#     """
#     Given an anchor image and a list of candidate image filenames (under IMAGES_DIR),
#     return a ranked list of candidates with compatibility scores.
#     """
#     # Encode anchor once
#     anchor_emb = encode_single_image(clip_model, preprocess, anchor_img)  # (1, D)

#     results = []
#     for rel_path in candidate_paths:
#         img_path = IMAGES_DIR / rel_path
#         if not img_path.exists():
#             continue
#         img = Image.open(img_path).convert("RGB")
#         cand_emb = encode_single_image(clip_model, preprocess, img)  # (1, D)

#         cos_sim = torch.nn.functional.cosine_similarity(anchor_emb, cand_emb).item()
#         with torch.no_grad():
#             logit = classifier(anchor_emb, cand_emb)
#             prob = torch.sigmoid(logit).item()

#         results.append(
#             {
#                 "path": rel_path,
#                 "cos_sim": float(cos_sim),
#                 "prob": float(prob),
#             }
#         )

#     # Sort by probability desc
#     results.sort(key=lambda x: x["prob"], reverse=True)
#     return results[:top_k]

def rank_candidates_for_anchor(
    clip_model,
    preprocess,
    classifier,
    anchor_img: Image.Image,
    candidate_paths: List[str],
    top_k: int = 8,
    desired_style: Optional[str] = None,
    prob_weight: float = 0.7,
    sim_weight: float = 0.2,
    style_weight: float = 0.1,
) -> List[Dict]:
    """
    Given an anchor image and a list of candidate image *relative paths*,
    rank candidates by a combined score:

        final_score = prob_weight * compat_prob
                    + sim_weight  * clip_cosine_similarity
                    + style_weight * style_alignment    (if desired_style is not None)

    Returns a list of dicts sorted by final_score descending, each with:
        {
            "path": str,
            "prob": float,
            "cos_sim": float,
            "style_score": Optional[float],
            "final_score": float,
        }
    """
    # 1) Encode anchor image once
    anchor_emb = encode_single_image(clip_model, preprocess, anchor_img)  # (1, D)

    # 2) If style is requested, prepare a text embedding for that style
    style_emb = None
    if desired_style is not None:
        style_key = desired_style.lower()
        if style_key in STYLE_PROMPTS:
            prompt = STYLE_PROMPTS[style_key]
        else:
            prompt = f"an outfit in {style_key} style"

        text_embs = _encode_text_prompts(clip_model, [prompt])  # (1, D)
        style_emb = text_embs[0:1]  # keep batch dim

    results = []

    with torch.no_grad():
        for rel_path in candidate_paths:
            img_path = IMAGES_DIR / rel_path
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            cand_emb = encode_single_image(clip_model, preprocess, img)  # (1, D)

            # CLIP cosine sim
            cos_sim = float((anchor_emb @ cand_emb.T).item())

            # Compatibility probability via classifier
            logit = classifier(anchor_emb, cand_emb)
            prob = float(torch.sigmoid(logit).item())

            # Optional style alignment
            style_score = None
            if style_emb is not None:
                style_score = float((cand_emb @ style_emb.T).item())

            # Final score
            final = prob_weight * prob + sim_weight * cos_sim
            if style_score is not None:
                final += style_weight * style_score

            results.append(
                {
                    "path": rel_path,
                    "prob": prob,
                    "cos_sim": cos_sim,
                    "style_score": style_score,
                    "final_score": final,
                }
            )

    # Sort by final_score descending
    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:top_k]

def score_outfit(
    clip_model,
    preprocess,
    classifier,
    images: List[Image.Image],
) -> Tuple[np.ndarray, float]:
    """
    Given a list of PIL images (length >= 2), compute:
      - pairwise probability matrix (NxN, symmetric, diag = 1)
      - overall outfit score (mean of upper-triangular probs)
    """
    n = len(images)
    assert n >= 2, "Need at least 2 items to score an outfit."

    # Encode all images once
    embs = []
    for img in images:
        embs.append(encode_single_image(clip_model, preprocess, img))
    embs = torch.cat(embs, dim=0)  # (N, D)

    # pairwise matrix
    probs = np.ones((n, n), dtype=np.float32)

    with torch.no_grad():
        for i in range(n):
            for j in range(i + 1, n):
                e1 = embs[i : i + 1]
                e2 = embs[j : j + 1]
                logit = classifier(e1, e2)
                p = torch.sigmoid(logit).item()
                probs[i, j] = p
                probs[j, i] = p

    # overall outfit score: mean of upper triangle excluding diagonal
    if n > 1:
        triu_indices = np.triu_indices(n, k=1)
        outfit_score = float(probs[triu_indices].mean())
    else:
        outfit_score = 1.0

    return probs, outfit_score

# ---------- Style & color analysis helpers ----------

# Simple global style prompts; you can tweak wording later.
STYLE_PROMPTS: Dict[str, str] = {
    "casual": "a casual everyday outfit",
    "formal": "a formal elegant outfit",
    "streetwear": "a streetwear style outfit",
    "sporty": "a sporty athletic outfit",
    "minimal": "a minimalist neutral-tone outfit",
    "party": "a party or going-out outfit",
}


def _encode_text_prompts(clip_model, prompts: List[str]) -> torch.Tensor:
    """
    Encode a list of text prompts using open_clip tokenizer + CLIP text encoder.
    Returns L2-normalized text embeddings of shape (N, D).
    """
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer(prompts).to(DEVICE)

    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)

    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb  # (N, D)


def analyze_style(
    clip_model,
    preprocess,
    img: Image.Image,
    style_prompts: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Given a single item image, compute similarity to a set of style prompts
    using CLIP image-text similarity.

    Returns a list of dicts:
        [{ "name": "casual", "score": 0.83, "prob": 0.29 }, ...]
    where:
        - score: cosine similarity (raw)
        - prob: softmax over all styles (rough 'percentage' of style)
    """
    if style_prompts is None:
        style_prompts = STYLE_PROMPTS

    # 1) Encode image
    img_emb = encode_single_image(clip_model, preprocess, img)  # (1, D)

    # 2) Encode text prompts
    names = list(style_prompts.keys())
    prompts = [style_prompts[n] for n in names]
    text_embs = _encode_text_prompts(clip_model, prompts)       # (K, D)

    # 3) Cosine similarity image vs each text prompt
    with torch.no_grad():
        sims = (img_emb @ text_embs.T).squeeze(0).cpu().numpy()  # (K,)

    # 4) Softmax over sims for nicer percentages
    # small temperature to spread distribution a bit
    temp = 0.07
    exps = np.exp(sims / temp)
    probs = exps / exps.sum()

    results = []
    for name, s, p in zip(names, sims, probs):
        results.append(
            {
                "name": name,
                "score": float(s),
                "prob": float(p),
            }
        )

    # sort by probability descending
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results


def extract_color_palette(
    img: Image.Image,
    n_colors: int = 3,
    sample_size: int = 5000,
) -> List[Dict]:
    """
    Extract a small color palette from the image using KMeans in RGB space.

    Returns a list of dicts:
        [{ "rgb": (r, g, b), "hex": "#rrggbb" }, ...]
    """
    # Resize for speed
    img_small = img.resize((128, 128))
    arr = np.array(img_small)  # (H, W, 3)
    arr = arr.reshape(-1, 3)

    # Optional subsample for speed
    if arr.shape[0] > sample_size:
        idx = np.random.choice(arr.shape[0], size=sample_size, replace=False)
        arr = arr[idx]

    # Run KMeans
    kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
    kmeans.fit(arr)
    centers = kmeans.cluster_centers_.astype(int)  # (n_colors, 3)

    palette = []
    for c in centers:
        r, g, b = [int(x) for x in c.tolist()]
        hex_code = "#{:02X}{:02X}{:02X}".format(r, g, b)
        palette.append({"rgb": (r, g, b), "hex": hex_code})

    return palette


# ---------- Ollama LLM helper ----------

def call_ollama(
    prompt: str,
    model: str = "llama3.1",
    url: str = "http://localhost:11434/api/chat",
    timeout: int = 30,
) -> Optional[str]:
    """
    Call a local Ollama model. Returns the response text or None on failure.

    Make sure `ollama serve` is running and the model is pulled, e.g.:
        ollama pull llama3.1
        ollama serve
    """
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        # We don't want the whole app to crash if Ollama isn't running.
        print(f"[Ollama] Error calling model: {e}")
        return None
