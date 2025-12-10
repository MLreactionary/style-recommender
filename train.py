# train.py
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm

from core import (
    DEVICE,
    PAIRS_CSV,
    MODELS_DIR,
    MODEL_PATH,
    PairDataset,
    CompatibilityMLP,
    load_clip_model,
    compute_batch_embeddings,
)

BATCH_SIZE = 32
NUM_EPOCHS = 5       
LR = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(preprocess):
    train_ds = PairDataset(split="train", preprocess=preprocess, csv_path=PAIRS_CSV)
    val_ds   = PairDataset(split="val",   preprocess=preprocess, csv_path=PAIRS_CSV)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, val_loader


def evaluate(clip_model, classifier, loader, criterion):
    classifier.eval()
    all_labels, all_probs = [], []
    total_loss, n_samples = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(DEVICE).float()
            emb1, emb2 = compute_batch_embeddings(clip_model, batch)
            logits = classifier(emb1, emb2)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, acc, f1, auc


def main():
    set_seed(RANDOM_SEED)
    print(f"Using device: {DEVICE}")

    clip_model, preprocess = load_clip_model()
    train_loader, val_loader = get_dataloaders(preprocess)

    embed_dim = clip_model.visual.output_dim
    classifier = CompatibilityMLP(embed_dim=embed_dim).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_auc = -1.0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        classifier.train()
        epoch_loss, n_train = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for batch in pbar:
            labels = batch["label"].to(DEVICE).float()
            emb1, emb2 = compute_batch_embeddings(clip_model, batch)

            logits = classifier(emb1, emb2)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / max(n_train, 1)
        print(f"\nEpoch {epoch} train loss: {avg_train_loss:.4f}")

        val_loss, val_acc, val_f1, val_auc = evaluate(
            clip_model, classifier, val_loader, criterion
        )
        print(
            f"Val loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.3f} | F1: {val_f1:.3f} | AUC: {val_auc:.3f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {"state_dict": classifier.state_dict(), "embed_dim": embed_dim},
                MODEL_PATH,
            )
            print(f"✅ Saved best model to {MODEL_PATH} (AUC={val_auc:.3f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
