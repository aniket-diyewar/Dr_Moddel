"""
train_clahe.py
Binary DR Detection — CLAHE Dataset
GPU Optimized for GTX 1650 (4GB VRAM)
AMP + cuDNN Benchmark + Mixed Precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import multiprocessing

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

# ─────────────────────────────────────────────
# GPU SETUP — GTX 1650 Optimized
# ─────────────────────────────────────────────

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True
    torch.backends.cudnn.benchmark         = True   # fastest conv algo
    torch.backends.cudnn.deterministic     = False
else:
    DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────
# CONFIG — tuned for 4GB VRAM
# ─────────────────────────────────────────────

DATASET_DIR   = "dataset_clahe"
BATCH_SIZE    = 32     # safe for 4GB — dont increase
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 50
STAGE1_LR     = 1e-3
STAGE2_LR     = 1e-5
PATIENCE      = 10
NUM_WORKERS   = 0
PIN_MEMORY    = False
NUM_CLASSES   = 2

CHECKPOINT_DIR = "checkpoints_clahe"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# AMP Scaler — speeds up GTX 1650 by ~40%
scaler = torch.amp.GradScaler(enabled=True)

# ── Auto-detect classes ──
_probe       = ImageFolder(root=os.path.join(DATASET_DIR, "train"))
CLASS_NAMES  = _probe.classes
CLASS_COUNTS = [len(os.listdir(os.path.join(DATASET_DIR, "train", c)))
                for c in CLASS_NAMES]

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
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
# CLASS WEIGHTS + SAMPLER
# ─────────────────────────────────────────────

total         = sum(CLASS_COUNTS)
class_weights = torch.tensor(
    [total / c for c in CLASS_COUNTS], dtype=torch.float32
).to(DEVICE)

def make_weighted_sampler(dataset):
    labels         = [dataset.targets[i] for i in range(len(dataset))]
    sample_weights = [total / CLASS_COUNTS[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights,
                                 num_samples=len(sample_weights),
                                 replacement=True)

# ─────────────────────────────────────────────
# FOCAL LOSS
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce  = F.cross_entropy(inputs, targets,
                              weight=self.alpha, reduction='none')
        pt  = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    )
    return model.to(DEVICE)

# ─────────────────────────────────────────────
# TRAIN ONE EPOCH — AMP Optimized
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # zero_grad set_to_none=True → faster memory release
        optimizer.zero_grad(set_to_none=True)

        # AMP — automatic mixed precision
        with torch.amp.autocast(device_type='cuda'):
            out  = model(imgs)
            loss = criterion(out, labels)

        # Scaled backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        n        += imgs.size(0)

    return loss_sum / n, correct / n

# ─────────────────────────────────────────────
# EVALUATE — AMP Optimized
# ─────────────────────────────────────────────

def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                out  = model(imgs)
                loss = criterion(out, labels)

            loss_sum += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            n        += imgs.size(0)

    return loss_sum / n, correct / n

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
        "scaler_state": scaler.state_dict(),
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
    if "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    print(f"  ▶ Resumed → Stage={ckpt['stage']} "
          f"Epoch={ckpt['epoch']} "
          f"BestLoss={ckpt['best_loss']:.4f}")
    return (ckpt["stage"], ckpt["epoch"],
            ckpt["best_loss"], ckpt["es_counter"])

# ─────────────────────────────────────────────
# STAGE 1 — Classifier Head
# ─────────────────────────────────────────────

def stage1_train(model, train_loader, val_loader, criterion):
    print("\n── STAGE 1: Training Classifier Head ──")
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=STAGE1_LR
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
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = evaluate(model, val_loader, criterion)
        print(f"  Epoch {ep:02d}/{STAGE1_EPOCHS} | "
              f"Train Loss: {tl:.4f}  Acc: {ta:.4f} | "
              f"Val Loss: {vl:.4f}  Acc: {va:.4f}")
        save_ckpt("stage1", ep, model, optimizer,
                  None, vl, 0, ckpt_path)
    return model

# ─────────────────────────────────────────────
# STAGE 2 — Fine-Tune Top 50%
# ─────────────────────────────────────────────

def stage2_train(model, train_loader, val_loader, criterion,
                 save_path="best_model_clahe.pth"):
    print("\n── STAGE 2: Fine-Tuning Top 50% of Backbone ──")

    layers = list(model.features.children())
    for i, layer in enumerate(layers):
        if i >= int(len(layers) * 0.5):
            for p in layer.parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"  Trainable params : {trainable:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR
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
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion)
        vl, va = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"  Epoch {ep:02d}/{STAGE2_EPOCHS} | "
              f"Train Loss: {tl:.4f}  Acc: {ta:.4f} | "
              f"Val Loss: {vl:.4f}  Acc: {va:.4f}")

        if vl < best_loss:
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
    print("\n══════════════════════════════════════════")
    print("   DR Detection — CLAHE + GTX 1650 Mode  ")
    print("══════════════════════════════════════════")

    # GPU Info
    props = torch.cuda.get_device_properties(0)
    vram  = props.total_memory / 1024**3
    print(f"\n  GPU    : {props.name}")
    print(f"  VRAM   : {vram:.1f} GB")
    print(f"  AMP    : Enabled (Mixed Precision)")
    print(f"  cuDNN  : Benchmark ON")
    print(f"  Batch  : {BATCH_SIZE} (safe for 4GB)")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Counts : {CLASS_COUNTS}")

    train_ds = ImageFolder(
        os.path.join(DATASET_DIR, "train"),
        transform=train_transform
    )
    val_ds = ImageFolder(
        os.path.join(DATASET_DIR, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_ds),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model     = build_model()
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    model = stage1_train(model, train_loader, val_loader, criterion)
    model = stage2_train(model, train_loader, val_loader, criterion,
                         save_path="best_model_clahe.pth")

    _, final_acc = evaluate(model, val_loader, criterion)
    print(f"\n  Final Val Accuracy : {final_acc:.4f}")
    torch.save(model.state_dict(), "trained_model_clahe.pth")
    print("  Model saved → trained_model_clahe.pth")
    return model


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_training()