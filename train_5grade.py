"""
train_5grade.py — FINAL FIXED VERSION
Fixes: NaN loss + model predicting single class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import multiprocessing

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from PIL import Image

# ─────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
else:
    DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR    = "dataset_5grade"
BATCH_SIZE     = 32
STAGE1_EPOCHS  = 15
STAGE2_EPOCHS  = 70
STAGE1_LR      = 1e-3
STAGE2_LR      = 1e-5
PATIENCE       = 12
NUM_WORKERS    = 0
PIN_MEMORY     = False
NUM_CLASSES    = 5
CHECKPOINT_DIR = "checkpoints_5grade"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Auto-detect ──
_probe       = ImageFolder(root=os.path.join(DATASET_DIR, "train"))
CLASS_NAMES  = _probe.classes
CLASS_COUNTS = [len(os.listdir(os.path.join(DATASET_DIR, "train", c)))
                for c in CLASS_NAMES]
MINORITY_IDX = [i for i, c in enumerate(CLASS_COUNTS) if c < 400]

# ─────────────────────────────────────────────
# CLASS WEIGHTS — inverse frequency (stronger)
# ─────────────────────────────────────────────

total         = sum(CLASS_COUNTS)
# Inverse frequency — stronger signal than sqrt
class_weights = torch.tensor(
    [total / (NUM_CLASSES * c) for c in CLASS_COUNTS],
    dtype=torch.float32
).to(DEVICE)

print("\n  Class weights (inverse freq):")
for name, count, w in zip(CLASS_NAMES, CLASS_COUNTS,
                          class_weights.cpu().tolist()):
    print(f"    {name:<25} count={count:>4}  weight={w:.3f}")

# ─────────────────────────────────────────────
# WEIGHTED SAMPLER — aggressive for minority
# ─────────────────────────────────────────────

def make_weighted_sampler(dataset):
    labels = [dataset.targets[i] for i in range(len(dataset))]
    # Pure inverse — every class gets equal total representation
    weights = [total / (NUM_CLASSES * CLASS_COUNTS[lbl])
                for lbl in labels]
    return WeightedRandomSampler(
        weights,
        num_samples = len(weights),
        replacement = True
    )

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────

standard_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

strong_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# PER-CLASS DATASET
# ─────────────────────────────────────────────

class PerClassDataset(Dataset):
    def __init__(self, root, minority_idx):
        self.base         = ImageFolder(root=root)
        self.minority_idx = set(minority_idx)
        self.targets      = self.base.targets
        self.classes      = self.base.classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        tf  = (strong_transform if label in self.minority_idx
               else standard_transform)
        return tf(img), label

# ─────────────────────────────────────────────
# LOSS — simple CrossEntropyLoss (stable)
# FocalLoss caused NaN with AMP + 5 classes
# ─────────────────────────────────────────────

def get_criterion():
    return nn.CrossEntropyLoss(
        weight     = class_weights,
        label_smoothing = 0.1   # helps with adjacent grade confusion
    )

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False

    num_features     = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.SiLU(),                    # SiLU better than ReLU for EfficientNet
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE)

# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        out  = model(imgs)
        loss = criterion(out, labels)

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        n        += imgs.size(0)

    return (loss_sum / n if n > 0 else 0.0), correct / n

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            out  = model(imgs)
            loss = criterion(out, labels)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss_sum += loss.item() * imgs.size(0)

            preds = out.argmax(1)
            correct     += (preds == labels).sum().item()
            n           += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy for monitoring
    per_class = {}
    for g in range(NUM_CLASSES):
        idx    = [i for i, l in enumerate(all_labels) if l == g]
        if idx:
            correct_g = sum(all_preds[i] == g for i in idx)
            per_class[CLASS_NAMES[g]] = correct_g / len(idx)

    return (loss_sum / n if n > 0 else 0.0), correct / n, per_class

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────

def save_ckpt(stage, epoch, model, optimizer,
              scheduler, best_loss, es_counter, path):
    torch.save({
        "stage"      : stage,
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler else None,
        "best_loss"  : best_loss,
        "es_counter" : es_counter,
    }, path)
    print(f"  💾 Checkpoint → {path}  (epoch={epoch})")

def load_ckpt(path, model, optimizer, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and ckpt["sched_state"]:
        scheduler.load_state_dict(ckpt["sched_state"])
    print(f"  ▶ Resumed → Stage={ckpt['stage']} "
          f"Epoch={ckpt['epoch']} "
          f"BestLoss={ckpt['best_loss']:.4f}")
    return (ckpt["stage"], ckpt["epoch"],
            ckpt["best_loss"], ckpt["es_counter"])

# ─────────────────────────────────────────────
# STAGE 1
# ─────────────────────────────────────────────

def stage1_train(model, train_loader, val_loader, criterion):
    print("\n── STAGE 1: Training Classifier Head ──")
    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=STAGE1_LR, weight_decay=1e-4
    )
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt_stage1.pth")

    start = 1
    if os.path.exists(ckpt_path):
        _, last, _, _ = load_ckpt(ckpt_path, model, optimizer)
        start = last + 1

    if start > STAGE1_EPOCHS:
        print("  Stage 1 complete — skipping.")
        return model

    for ep in range(start, STAGE1_EPOCHS + 1):
        tl, ta       = train_one_epoch(model, train_loader,
                                       optimizer, criterion)
        vl, va, pca  = evaluate(model, val_loader, criterion)

        print(f"  Epoch {ep:02d}/{STAGE1_EPOCHS} | "
              f"Train Loss: {tl:.4f}  Acc: {ta:.4f} | "
              f"Val Loss: {vl:.4f}  Acc: {va:.4f}")

        # Print per-class val accuracy
        for cls, acc in pca.items():
            short = cls.split('_')[0] + '_' + cls.split('_')[1]
            print(f"    {short:<20} {acc*100:.1f}%")

        save_ckpt("stage1", ep, model, optimizer,
                  None, vl, 0, ckpt_path)
    return model

# ─────────────────────────────────────────────
# STAGE 2
# ─────────────────────────────────────────────

def stage2_train(model, train_loader, val_loader, criterion,
                 save_path="best_model_5grade.pth"):
    print("\n── STAGE 2: Fine-Tuning Top 50% of Backbone ──")

    layers = list(model.features.children())
    for i, layer in enumerate(layers):
        if i >= int(len(layers) * 0.5):
            for p in layer.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-7
    )
    ckpt_path  = os.path.join(CHECKPOINT_DIR, "ckpt_stage2.pth")
    start      = 1
    best_loss  = float('inf')
    es_counter = 0

    if os.path.exists(ckpt_path):
        _, last, best_loss, es_counter = load_ckpt(
            ckpt_path, model, optimizer, scheduler)
        start = last + 1

    if start > STAGE2_EPOCHS:
        print("  Stage 2 complete — skipping.")
        model.load_state_dict(
            torch.load(save_path, weights_only=True))
        return model

    for ep in range(start, STAGE2_EPOCHS + 1):
        tl, ta      = train_one_epoch(model, train_loader,
                                      optimizer, criterion)
        vl, va, pca = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"  Epoch {ep:02d}/{STAGE2_EPOCHS} | "
              f"Train Loss: {tl:.4f}  Acc: {ta:.4f} | "
              f"Val Loss: {vl:.4f}  Acc: {va:.4f}")

        # Per-class accuracy every 5 epochs
        if ep % 5 == 0:
            print("  Per-class val accuracy:")
            for cls, acc in pca.items():
                bar   = '█' * int(acc * 20)
                short = cls.replace('Grade','G').replace('_','')
                print(f"    {short:<18} {bar:<20} {acc*100:.1f}%")

        if vl < best_loss and vl > 0:
            best_loss  = vl
            es_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Best saved → {save_path} "
                  f"(val loss: {best_loss:.4f})")
        else:
            es_counter += 1
            print(f"  No improvement. "
                  f"Counter: {es_counter}/{PATIENCE}")

        save_ckpt("stage2", ep, model, optimizer, scheduler,
                  best_loss, es_counter, ckpt_path)

        if es_counter >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {ep}")
            break

    model.load_state_dict(
        torch.load(save_path, weights_only=True))
    print(f"  ✅ Best val loss: {best_loss:.4f}")
    return model

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_training():
    print("\n══════════════════════════════════════════════")
    print("   5-Grade DR Classification — FIXED         ")
    print("══════════════════════════════════════════════")

    props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory/1024**3:.1f} GB")
    print(f"  Loss    : CrossEntropy + label_smoothing=0.1")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"\n  {'Class':<25} {'Count':>6} {'Weight':>8} {'Aug':>10}")
    print(f"  {'─'*56}")
    for i, (nm, cnt) in enumerate(zip(CLASS_NAMES, CLASS_COUNTS)):
        w   = class_weights[i].item()
        aug = "STRONG" if i in MINORITY_IDX else "standard"
        print(f"  {nm:<25} {cnt:>6} {w:>8.3f}   {aug}")

    train_ds = PerClassDataset(
        root=os.path.join(DATASET_DIR, "train"),
        minority_idx=MINORITY_IDX
    )
    val_ds = ImageFolder(
        root=os.path.join(DATASET_DIR, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_ds),
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model     = build_model()
    criterion = get_criterion()

    model = stage1_train(model, train_loader, val_loader, criterion)
    model = stage2_train(model, train_loader, val_loader, criterion,
                         save_path="best_model_5grade.pth")

    _, final_acc, final_pca = evaluate(model, val_loader, criterion)
    print(f"\n  Final Val Accuracy : {final_acc:.4f}")
    print("\n  Final Per-Class Accuracy:")
    for cls, acc in final_pca.items():
        status = "✅" if acc >= 0.7 else "⚠️" if acc >= 0.5 else "❌"
        print(f"    {status} {cls:<25} {acc*100:.1f}%")

    torch.save(model.state_dict(), "trained_model_5grade.pth")
    print("\n  Model saved → trained_model_5grade.pth")
    return model


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_training()