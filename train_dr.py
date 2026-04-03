"""
train_dr.py
Binary DR Detection — APTOS 2019
EfficientNet-B0 | No_DR vs DR | Two-Stage Transfer Learning
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
from torch.utils.data import DataLoader, WeightedRandomSampler

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR   = "dataset"
BATCH_SIZE    = 32
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 50
STAGE1_LR     = 1e-3
STAGE2_LR     = 1e-5
PATIENCE      = 10
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS   = 0
PIN_MEMORY    = False
NUM_CLASSES   = 2

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Auto-detect class counts ──
_probe       = ImageFolder(root=os.path.join(DATASET_DIR, "train"))
CLASS_NAMES  = _probe.classes        # ['DR', 'No_DR']
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
        ce_loss = F.cross_entropy(inputs, targets,
                                  weight=self.alpha,
                                  reduction='none')
        pt      = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_features     = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model.to(DEVICE)


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss  += loss.item() * images.size(0)
        correct       += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)

    return running_loss / total_samples, correct / total_samples


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss  += loss.item() * images.size(0)
            correct       += (outputs.argmax(1) == labels).sum().item()
            total_samples += images.size(0)

    return running_loss / total_samples, correct / total_samples


# ─────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────

def save_checkpoint(stage, epoch, model, optimizer,
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
    print(f"  💾 Checkpoint saved → {path}  (stage={stage}, epoch={epoch})")


def load_checkpoint(path, model, optimizer, scheduler=None):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    if scheduler and ckpt["sched_state"]:
        scheduler.load_state_dict(ckpt["sched_state"])
    print(f"  ▶ Resumed → Stage={ckpt['stage']}  "
          f"Epoch={ckpt['epoch']}  "
          f"Best loss={ckpt['best_loss']:.4f}")
    return (ckpt["stage"], ckpt["epoch"],
            ckpt["best_loss"], ckpt["es_counter"])


# ─────────────────────────────────────────────
# STAGE 1 — Classifier head only
# ─────────────────────────────────────────────

def stage1_train(model, train_loader, val_loader, criterion):
    print("\n── STAGE 1: Training Classifier Head ──")
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=STAGE1_LR)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt_stage1.pth")

    start_epoch = 1
    if os.path.exists(ckpt_path):
        _, last_epoch, _, _ = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = last_epoch + 1

    if start_epoch > STAGE1_EPOCHS:
        print("  Stage 1 already complete — skipping.")
        return model

    for epoch in range(start_epoch, STAGE1_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        print(f"  Epoch {epoch:02d}/{STAGE1_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")
        save_checkpoint("stage1", epoch, model, optimizer,
                        None, vl_loss, 0, ckpt_path)
    return model


# ─────────────────────────────────────────────
# STAGE 2 — Fine-tune top 50% backbone
# ─────────────────────────────────────────────

def stage2_train(model, train_loader, val_loader, criterion,
                 save_path="best_model_dr.pth"):
    print("\n── STAGE 2: Fine-Tuning Top 50% of Backbone ──")

    layers        = list(model.features.children())
    unfreeze_from = int(len(layers) * 0.5)
    for i, layer in enumerate(layers):
        if i >= unfreeze_from:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-7
    )
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ckpt_stage2.pth")

    start_epoch = 1
    best_loss   = float('inf')
    es_counter  = 0

    if os.path.exists(ckpt_path):
        _, last_epoch, best_loss, es_counter = load_checkpoint(
            ckpt_path, model, optimizer, scheduler)
        start_epoch = last_epoch + 1

    if start_epoch > STAGE2_EPOCHS:
        print("  Stage 2 complete — skipping.")
        model.load_state_dict(torch.load(save_path, weights_only=True))
        return model

    for epoch in range(start_epoch, STAGE2_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"  Epoch {epoch:02d}/{STAGE2_EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}")

        if vl_loss < best_loss:
            best_loss  = vl_loss
            es_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Best model saved → {save_path}  "
                  f"(val loss: {best_loss:.4f})")
        else:
            es_counter += 1
            print(f"  No improvement. "
                  f"Early stop counter: {es_counter}/{PATIENCE}")

        save_checkpoint("stage2", epoch, model, optimizer,
                        scheduler, best_loss, es_counter, ckpt_path)

        if es_counter >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"  ✅ Best val loss: {best_loss:.4f}")
    return model


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_training():
    print("\n══════════════════════════════════════")
    print("   DR Binary Detection — APTOS 2019   ")
    print("══════════════════════════════════════")
    print(f"  Classes : {CLASS_NAMES}")
    print(f"  Counts  : {CLASS_COUNTS}  (train)")

    train_dataset = ImageFolder(
        root=os.path.join(DATASET_DIR, "train"),
        transform=train_transform
    )
    val_dataset = ImageFolder(
        root=os.path.join(DATASET_DIR, "val"),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_dataset),
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model     = build_model()
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    model = stage1_train(model, train_loader, val_loader, criterion)
    model = stage2_train(model, train_loader, val_loader, criterion,
                         save_path="best_model_dr.pth")

    _, final_acc = evaluate(model, val_loader, criterion)
    print(f"\n  Final Val Accuracy : {final_acc:.4f}")
    torch.save(model.state_dict(), "trained_model_dr.pth")
    print("  Model saved → trained_model_dr.pth")
    return model


if __name__ == "__main__":
    multiprocessing.freeze_support()
    print(f"Using device: {DEVICE}")
    run_training()