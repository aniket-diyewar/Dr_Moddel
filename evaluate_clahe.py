"""
evaluate_clahe.py
Test Set Evaluation — CLAHE Model vs Original Model
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR = "dataset_clahe"
MODEL_PATH  = "best_model_clahe.pth"
BATCH_SIZE  = 32
NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY  = False

_probe      = ImageFolder(root=os.path.join(DATASET_DIR, "test"))
CLASS_NAMES = _probe.classes

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH,
                          map_location=DEVICE, weights_only=True))
    print(f"  ✅ Model loaded: {MODEL_PATH}")
    return model.to(DEVICE).eval()

# ─────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────

def get_predictions(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs    = imgs.to(DEVICE)
            with torch.amp.autocast(device_type='cuda'):
                outputs = torch.softmax(model(imgs), dim=1)
            preds   = outputs.argmax(1).cpu().numpy()
            probs   = outputs[:, 0].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))

# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(y_true, y_pred, y_probs):
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, 1 - y_probs)

    tn, fp, fn, tp = cm.ravel()
    sensitivity    = tp / (tp + fn)
    specificity    = tn / (tn + fp)

    print("\n" + "═"*60)
    print("  CLAHE MODEL — CLASSIFICATION REPORT")
    print("═"*60)
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, digits=4))
    print("═"*60)
    print(f"  Test Accuracy  : {acc*100:.2f}%")
    print(f"  ROC-AUC Score  : {auc:.4f}")
    print(f"  Sensitivity    : {sensitivity*100:.2f}%")
    print(f"  Specificity    : {specificity*100:.2f}%")
    print(f"\n  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print("═"*60)

    # ── Comparison Table ──
    print("\n" + "═"*60)
    print("  COMPARISON — Original vs CLAHE")
    print("═"*60)
    print(f"  {'Metric':<20} {'Original':>10} {'CLAHE':>10} {'Change':>10}")
    print(f"  {'-'*52}")

    orig = {
        "Test Accuracy" : 97.73,
        "ROC-AUC"       : 0.9957,
        "Sensitivity"   : 99.07,
        "Specificity"   : 96.44,
    }
    clahe = {
        "Test Accuracy" : acc*100,
        "ROC-AUC"       : auc,
        "Sensitivity"   : sensitivity*100,
        "Specificity"   : specificity*100,
    }

    for metric in orig:
        o = orig[metric]
        c = clahe[metric]
        diff = c - o
        arrow = "⬆" if diff > 0 else "⬇"
        color = "+" if diff > 0 else ""
        print(f"  {metric:<20} {o:>10.2f} {c:>10.2f} "
              f"   {arrow} {color}{diff:.2f}")

    print("═"*60)
    return acc, auc, sensitivity, specificity

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

def plot_results(y_true, y_pred, y_probs,
                 acc, auc, sens, spec):

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.patch.set_facecolor('#03060f')
    for ax in axes:
        ax.set_facecolor('#03060f')

    fig.suptitle("CLAHE Model — Test Set Evaluation",
                 fontsize=14, color='white', y=1.01)

    # ── Confusion Matrix ──
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f',
                cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=axes[0], vmin=0, vmax=1,
                annot_kws={"size":14, "color":"white"})
    axes[0].set_title('Confusion Matrix',
                      color='white', fontsize=12)
    axes[0].set_xlabel('Predicted', color='#64748b')
    axes[0].set_ylabel('True',      color='#64748b')
    axes[0].tick_params(colors='#64748b')

    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_true, 1 - y_probs, pos_label=0)
    axes[1].plot(fpr, tpr, color='#00b4ff', lw=2.5,
                 label=f'CLAHE AUC = {auc:.4f}')
    axes[1].axhline(0.9957, color='#fb923c', lw=1.5,
                    linestyle='--', label='Original AUC = 0.9957')
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='#00b4ff')
    axes[1].plot([0,1],[0,1],'w--', alpha=0.3)
    axes[1].set_xlabel('False Positive Rate', color='#64748b')
    axes[1].set_ylabel('True Positive Rate',  color='#64748b')
    axes[1].set_title('ROC Curve — CLAHE vs Original',
                      color='white', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(colors='#64748b')
    axes[1].set_facecolor('#03060f')
    axes[1].spines['bottom'].set_color('#334155')
    axes[1].spines['left'].set_color('#334155')
    axes[1].spines['top'].set_color('#334155')
    axes[1].spines['right'].set_color('#334155')
    axes[1].grid(alpha=0.1, color='white')

    # ── Comparison Bar ──
    metrics  = ['Accuracy', 'AUC×100', 'Sensitivity', 'Specificity']
    original = [97.73, 99.57, 99.07, 96.44]
    clahe_v  = [acc*100, auc*100, sens*100, spec*100]

    x     = np.arange(len(metrics))
    width = 0.35

    b1 = axes[2].bar(x - width/2, original, width,
                     label='Original', color='#fb923c',
                     alpha=0.8, edgecolor='white', linewidth=0.5)
    b2 = axes[2].bar(x + width/2, clahe_v, width,
                     label='CLAHE', color='#00b4ff',
                     alpha=0.8, edgecolor='white', linewidth=0.5)

    for bar in list(b1) + list(b2):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.2,
                     f'{bar.get_height():.1f}',
                     ha='center', va='bottom',
                     fontsize=8, color='white')

    axes[2].set_ylim(90, 102)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(metrics, fontsize=9, color='#64748b')
    axes[2].set_title('Original vs CLAHE Comparison',
                      color='white', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].set_facecolor('#03060f')
    axes[2].tick_params(colors='#64748b')
    axes[2].spines['bottom'].set_color('#334155')
    axes[2].spines['left'].set_color('#334155')
    axes[2].spines['top'].set_color('#334155')
    axes[2].spines['right'].set_color('#334155')
    axes[2].grid(alpha=0.1, color='white', axis='y')

    plt.tight_layout()
    plt.savefig("clahe_evaluation.png", dpi=130,
                bbox_inches='tight', facecolor='#03060f')
    plt.show()
    print("  📊 Saved → clahe_evaluation.png")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("\n══════════════════════════════════════")
    print("   CLAHE Model — Test Set Evaluation  ")
    print("══════════════════════════════════════")

    test_ds = ImageFolder(
        root=os.path.join(DATASET_DIR, "test"),
        transform=test_transform
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    print(f"  Test images : {len(test_ds)}")
    print(f"  Classes     : {CLASS_NAMES}")

    model              = load_model()
    y_true, y_pred, y_probs = get_predictions(model, test_loader)
    acc, auc, sens, spec    = print_report(y_true, y_pred, y_probs)
    plot_results(y_true, y_pred, y_probs, acc, auc, sens, spec)

    print("\n  ✅ Evaluation complete!")