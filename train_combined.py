"""
train_combined.py  —  v2
5-Grade DR: APTOS + Messidor-2 + IDRiD
EfficientNet-B3 + OrdinalLoss + GTX 1650

KEY FIXES vs v1:
  1. Save best model on val ACCURACY (not val loss)
     → In v1, ep1 of Stage2 (loss=1.989) was saved forever,
       even though ep10 reached 66.6% acc
  2. BalancedDataset caps majority classes + oversamples minority
     → G1_Mild was chronically starved; now every class gets ~500/epoch
  3. Stage2 LR raised 1e-5 → 3e-5, unfreezes top 65% backbone
  4. Patience raised to 15, prints QWK every epoch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import multiprocessing

from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
from sklearn.metrics import cohen_kappa_score

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

DATASET_DIR    = "dataset_combined"
BATCH_SIZE     = 16
STAGE1_EPOCHS  = 15
STAGE2_EPOCHS  = 70
STAGE1_LR      = 1e-3
STAGE2_LR      = 3e-5        # FIX: was 1e-5, too slow to learn
BACKBONE_UNFREEZE = 0.65     # FIX: was 0.50, now unfreezes top 65%
PATIENCE       = 15          # FIX: was 12
NUM_WORKERS    = 0
PIN_MEMORY     = False
NUM_CLASSES    = 5
CHECKPOINT_DIR = "checkpoints_combined"
SAVE_PATH      = "best_model_combined.pth"
TARGET_PER_CLASS = 500       # FIX: balance target (caps majority, oversamps minority)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Auto-detect class info ──
_probe       = ImageFolder(root=os.path.join(DATASET_DIR, "train"))
CLASS_NAMES  = _probe.classes
CLASS_COUNTS = [
    len(os.listdir(os.path.join(DATASET_DIR, "train", c)))
    for c in CLASS_NAMES
]
total         = sum(CLASS_COUNTS)
class_weights = torch.tensor(
    [total / (NUM_CLASSES * c) for c in CLASS_COUNTS],
    dtype=torch.float32
).to(DEVICE)

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────

standard_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(300, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

strong_tf = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=35),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.05),
    transforms.RandomResizedCrop(300, scale=(0.65, 1.0)),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# BALANCED DATASET (FIX 2)
# Caps majority classes, oversamples minority
# so every class contributes ~TARGET_PER_CLASS
# samples per epoch — no file deletion needed
# ─────────────────────────────────────────────

class BalancedDataset(Dataset):
    """
    Builds a balanced sample list each time reset() is called.
    Majority classes: randomly subsample to TARGET_PER_CLASS.
    Minority classes: repeat with replacement to TARGET_PER_CLASS.
    Each repeat uses a different random augmentation → no duplicate views.
    """
    def __init__(self, root, target_per_class=TARGET_PER_CLASS):
        self.base             = ImageFolder(root=root)
        self.target           = target_per_class
        self.class_names      = self.base.classes
        self.num_classes      = len(self.class_names)

        # Group sample indices by class
        self.class_samples = {i: [] for i in range(self.num_classes)}
        for idx, (_, label) in enumerate(self.base.samples):
            self.class_samples[label].append(idx)

        self.samples = []   # filled by reset()
        self.targets = []
        self.reset()

    def reset(self):
        """Resample every epoch for fresh augmentation variety."""
        self.samples = []
        self.targets = []
        for cls, idxs in self.class_samples.items():
            if len(idxs) >= self.target:
                # Majority: subsample without replacement
                chosen = random.sample(idxs, self.target)
            else:
                # Minority: repeat with replacement
                chosen = random.choices(idxs, k=self.target)
            self.samples.extend(chosen)
            self.targets.extend([cls] * self.target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_idx = self.samples[idx]
        label    = self.targets[idx]
        path     = self.base.samples[base_idx][0]
        img      = Image.open(path).convert("RGB")
        # Use strong augmentation for minority classes
        n        = len(self.class_samples[label])
        tf       = strong_tf if n < self.target else standard_tf
        return tf(img), label

# ─────────────────────────────────────────────
# ORDINAL LOSS
# ─────────────────────────────────────────────

class OrdinalLoss(nn.Module):
    """
    CrossEntropy + L1 penalty on expected grade distance.
    Penalizes predicting Grade 0 when true is Grade 4
    more than predicting Grade 3.
    """
    def __init__(self, num_classes=5, alpha=0.5, weights=None):
        super().__init__()
        self.alpha = alpha
        self.n     = num_classes
        self.ce    = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    def forward(self, x, y):
        ce   = self.ce(x, y)
        prob = torch.softmax(x, dim=1)
        grades = torch.arange(self.n, dtype=torch.float32, device=x.device)
        exp  = (prob * grades).sum(1)
        ord_ = F.l1_loss(exp, y.float())
        return ce + self.alpha * ord_

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model():
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    feats = m.classifier[1].in_features  # 1536
    m.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(feats, 512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    return m.to(DEVICE)

# ─────────────────────────────────────────────
# METRICS — Quadratic Weighted Kappa
# ─────────────────────────────────────────────

def compute_qwk(preds, labels):
    try:
        return cohen_kappa_score(labels, preds, weights="quadratic")
    except Exception:
        return 0.0

def print_per_class(pca, prefix=""):
    for cls, acc in pca.items():
        s   = "✅" if acc >= 0.70 else "⚠️" if acc >= 0.50 else "❌"
        bar = "█" * int(acc * 20)
        print(f"  {prefix}{s} {cls.replace('Grade','G'):<22} {bar:<20} {acc*100:.1f}%")

# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────

def train_epoch(model, loader, opt, criterion):
    model.train()
    ls, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out  = model(imgs)
        loss = criterion(out, labels)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ls      += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n       += imgs.size(0)
    return ls / max(n, 1), correct / max(n, 1)

def evaluate(model, loader, criterion):
    model.eval()
    ls, correct, n  = 0.0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            out  = model(imgs)
            loss = criterion(out, labels)
            if not torch.isnan(loss):
                ls += loss.item() * imgs.size(0)
            p = out.argmax(1)
            correct     += (p == labels).sum().item()
            n           += imgs.size(0)
            preds_all.extend(p.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    pca = {}
    for g in range(NUM_CLASSES):
        idx = [i for i, l in enumerate(labels_all) if l == g]
        if idx:
            pca[CLASS_NAMES[g]] = sum(preds_all[i] == g for i in idx) / len(idx)

    qwk = compute_qwk(preds_all, labels_all)
    return ls / max(n, 1), correct / max(n, 1), pca, qwk

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────

def save_ckpt(stage, epoch, model, opt, sched, best_loss, best_acc, es, path):
    torch.save({
        "stage": stage, "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": opt.state_dict(),
        "sched_state": sched.state_dict() if sched else None,
        "best_loss": best_loss, "best_acc": best_acc, "es_counter": es
    }, path)
    print(f"  💾 Checkpoint → {path}  (epoch={epoch})")

def load_ckpt(path, model, opt, sched=None):
    c = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(c["model_state"])
    opt.load_state_dict(c["optim_state"])
    if sched and c.get("sched_state"):
        sched.load_state_dict(c["sched_state"])
    best_acc = c.get("best_acc", 0.0)
    print(f"  ▶ Resumed → Epoch={c['epoch']}  BestAcc={best_acc:.4f}")
    return c["stage"], c["epoch"], c.get("best_loss", float("inf")), best_acc, c["es_counter"]

# ─────────────────────────────────────────────
# STAGE 1 — Train head only
# ─────────────────────────────────────────────

def stage1(model, train_ds, vl, crit):
    print("\n── STAGE 1: Classifier Head ──")
    print(f"  LR={STAGE1_LR}  Epochs={STAGE1_EPOCHS}")

    opt   = torch.optim.Adam(model.classifier.parameters(), lr=STAGE1_LR, weight_decay=1e-4)
    path  = os.path.join(CHECKPOINT_DIR, "ckpt_s1.pth")
    start = 1
    best_loss = float("inf")
    best_acc  = 0.0
    es        = 0

    if os.path.exists(path):
        _, last, best_loss, best_acc, es = load_ckpt(path, model, opt)
        start = last + 1
    if start > STAGE1_EPOCHS:
        print("  Stage 1 complete — skipping.")
        return model

    for ep in range(start, STAGE1_EPOCHS + 1):
        train_ds.reset()   # reshuffle balanced samples each epoch
        tl_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        tls, ta         = train_epoch(model, tl_loader, opt, crit)
        vls, va, pca, qwk = evaluate(model, vl, crit)

        print(f"  Ep {ep:02d}/{STAGE1_EPOCHS} | "
              f"TrL:{tls:.3f} Acc:{ta:.3f} | "
              f"VaL:{vls:.3f} Acc:{va:.3f}  QWK:{qwk:.3f}")

        if va > best_acc:   # FIX: save on accuracy
            best_acc  = va
            best_loss = vls
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ⭐ New best acc {best_acc:.4f} — saved")

        print_per_class(pca)
        save_ckpt("s1", ep, model, opt, None, best_loss, best_acc, es, path)

    return model

# ─────────────────────────────────────────────
# STAGE 2 — Fine-tune top 65% backbone
# ─────────────────────────────────────────────

def stage2(model, train_ds, vl, crit):
    print(f"\n── STAGE 2: Fine-Tune Top {int(BACKBONE_UNFREEZE*100)}% Backbone ──")
    print(f"  LR={STAGE2_LR}  Patience={PATIENCE}  Epochs={STAGE2_EPOCHS}")

    # Unfreeze top BACKBONE_UNFREEZE fraction of backbone layers
    layers = list(model.features.children())
    cutoff = int(len(layers) * (1 - BACKBONE_UNFREEZE))
    for i, layer in enumerate(layers):
        for p in layer.parameters():
            p.requires_grad = (i >= cutoff)
    # Always keep classifier trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR, weight_decay=1e-4
    )
    # Warmup for 3 epochs, then cosine decay
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=3
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=STAGE2_EPOCHS - 3, eta_min=1e-7
    )
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup_sched, cosine_sched], milestones=[3]
    )

    path  = os.path.join(CHECKPOINT_DIR, "ckpt_s2.pth")
    start = 1
    best_loss = float("inf")
    best_acc  = 0.0
    es        = 0

    if os.path.exists(path):
        _, last, best_loss, best_acc, es = load_ckpt(path, model, opt, sched)
        start = last + 1
    if start > STAGE2_EPOCHS:
        print("  Stage 2 complete — skipping.")
        model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
        return model

    for ep in range(start, STAGE2_EPOCHS + 1):
        train_ds.reset()
        tl_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        tls, ta          = train_epoch(model, tl_loader, opt, crit)
        vls, va, pca, qwk = evaluate(model, vl, crit)
        current_lr = opt.param_groups[0]["lr"]
        sched.step()

        print(f"  Ep {ep:02d}/{STAGE2_EPOCHS} | "
              f"TrL:{tls:.3f} Acc:{ta:.3f} | "
              f"VaL:{vls:.3f} Acc:{va:.3f}  QWK:{qwk:.3f}  LR:{current_lr:.2e}")

        # FIX: save on val ACCURACY not val loss
        if va > best_acc:
            best_acc  = va
            best_loss = vls
            es        = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ⭐ New best acc {best_acc:.4f} — saved")
        else:
            es += 1
            print(f"  No improvement. Counter: {es}/{PATIENCE}")

        if ep % 3 == 0 or es == 0:
            print_per_class(pca, "    ")

        save_ckpt("s2", ep, model, opt, sched, best_loss, best_acc, es, path)

        if es >= PATIENCE:
            print(f"  ⏹ Early stop at epoch {ep}")
            break

    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    print(f"\n  ✅ Best val accuracy: {best_acc:.4f}")
    return model

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run():
    print("\n" + "═" * 56)
    print("  5-Grade DR — B3 v2  (APTOS + Messidor-2 + IDRiD)")
    print("═" * 56)

    props = torch.cuda.get_device_properties(0)
    print(f"\n  GPU   : {props.name}")
    print(f"  VRAM  : {props.total_memory/1024**3:.1f} GB")
    print(f"  Batch : {BATCH_SIZE}   Image: 300×300")
    print(f"  Target per class: {TARGET_PER_CLASS} (balanced)")

    print(f"\n  {'Class':<26} {'Raw':>5} {'Wt':>5}  Aug")
    print("  " + "─" * 48)
    for i, (nm, cnt) in enumerate(zip(CLASS_NAMES, CLASS_COUNTS)):
        w   = class_weights[i].item()
        aug = "STRONG (minority)" if cnt < TARGET_PER_CLASS else "standard"
        print(f"  {nm:<26} {cnt:>5} {w:>5.2f}  {aug}")
    balanced_total = TARGET_PER_CLASS * NUM_CLASSES
    print(f"  {'Balanced total/epoch:':<26} {balanced_total:>5}")

    # Datasets
    train_ds = BalancedDataset(os.path.join(DATASET_DIR, "train"), TARGET_PER_CLASS)
    val_ds   = ImageFolder(os.path.join(DATASET_DIR, "val"), transform=val_tf)

    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    model = build_model()
    crit  = OrdinalLoss(NUM_CLASSES, alpha=0.5, weights=class_weights)

    model = stage1(model, train_ds, val_loader, crit)
    model = stage2(model, train_ds, val_loader, crit)

    # Final evaluation
    _, acc, pca, qwk = evaluate(model, val_loader, crit)
    print(f"\n{'═'*56}")
    print(f"  FINAL  Val Acc: {acc:.4f}   QWK: {qwk:.4f}")
    print(f"{'═'*56}")
    print_per_class(pca)

    torch.save(model.state_dict(), "trained_model_combined.pth")
    print(f"\n  Saved → trained_model_combined.pth")
    print("  Next: py evaluate_combined.py")
    print("═" * 56)
    return model


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()