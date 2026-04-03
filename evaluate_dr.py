"""
evaluate_dr.py
Binary DR Evaluation — Test Set
Accuracy | AUC | Sensitivity | Specificity | Confusion Matrix
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

DATASET_DIR = "dataset"
MODEL_PATH  = "best_model_dr.pth"
BATCH_SIZE  = 32
NUM_CLASSES = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY  = False

# ── Auto-detect class names ──
_probe      = ImageFolder(root=os.path.join(DATASET_DIR, "test"))
CLASS_NAMES = _probe.classes   # ['DR', 'No_DR']

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
    num_features     = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH,
                          map_location=DEVICE, weights_only=True))
    print(f"  ✅ Model loaded from {MODEL_PATH}")
    return model.to(DEVICE).eval()


# ─────────────────────────────────────────────
# GET PREDICTIONS
# ─────────────────────────────────────────────

def get_predictions(model, loader):
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(DEVICE)
            outputs = torch.softmax(model(images), dim=1)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            # DR is index 0 — probability of DR
            probs   = outputs[:, 0].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


# ─────────────────────────────────────────────
# PRINT FULL REPORT
# ─────────────────────────────────────────────

def print_report(y_true, y_pred, y_probs):
    cm          = confusion_matrix(y_true, y_pred)
    acc         = accuracy_score(y_true, y_pred)
    auc         = roc_auc_score(y_true, 1 - y_probs)


    # DR = class index 0
    # TN=No_DR correct, FP=No_DR predicted as DR
    # FN=DR predicted as No_DR, TP=DR correct
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)   # DR recall
    specificity = tn / (tn + fp)   # No_DR recall

    print("\n" + "═"*55)
    print("  CLASSIFICATION REPORT")
    print("═"*55)
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES,
                                digits=4))
    print("═"*55)
    print(f"  ✅ Test Accuracy   : {acc*100:.2f}%")
    print(f"  ✅ ROC-AUC Score   : {auc:.4f}")
    print(f"  ✅ Sensitivity     : {sensitivity*100:.2f}%  "
          f"(DR correctly detected)")
    print(f"  ✅ Specificity     : {specificity*100:.2f}%  "
          f"(No_DR correctly identified)")
    print(f"\n  Raw Counts:")
    print(f"     True  DR  detected as DR  (TP) : {tp}")
    print(f"     True  DR  missed as No_DR (FN) : {fn}")
    print(f"     True  No_DR as No_DR      (TN) : {tn}")
    print(f"     True  No_DR as DR         (FP) : {fp}")
    print("═"*55)

    return acc, auc, sensitivity, specificity


# ─────────────────────────────────────────────
# PLOT RESULTS
# ─────────────────────────────────────────────

def plot_results(y_true, y_pred, y_probs,
                 acc, auc, sensitivity, specificity):

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("DR Binary Detection — Test Set Evaluation",
                 fontsize=15, fontweight='bold', y=1.02)

    # ── 1. Confusion Matrix ──
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES,
                ax=axes[0], vmin=0, vmax=1,
                annot_kws={"size": 14})
    axes[0].set_title('Confusion Matrix (Normalised)',
                      fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label',      fontsize=11)

    # ── 2. ROC Curve ──
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=0)
    axes[1].plot(fpr, tpr, color='#2ecc71', lw=2.5,
                 label=f'AUC = {auc:.4f}')
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='#2ecc71')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate',  fontsize=11)
    axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    # ── 3. Key Metrics Bar Chart ──
    metrics = ['Accuracy', 'ROC-AUC', 'Sensitivity', 'Specificity']
    values  = [acc, auc, sensitivity, specificity]
    colors  = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71']

    bars = axes[2].bar(metrics, [v * 100 for v in values],
                       color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f'{val*100:.2f}%',
                     ha='center', va='bottom',
                     fontsize=11, fontweight='bold')

    axes[2].axhline(y=90, color='green', linestyle='--',
                    alpha=0.5, label='90% line')
    axes[2].set_ylim(0, 115)
    axes[2].set_title('Key Metrics Summary',
                      fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Score (%)', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig("dr_evaluation.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  📊 Saved → dr_evaluation.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("\n══════════════════════════════════════")
    print("   DR Binary Evaluation — Test Set    ")
    print("══════════════════════════════════════")
    print(f"  Classes     : {CLASS_NAMES}")

    test_dataset = ImageFolder(
        root=os.path.join(DATASET_DIR, "test"),
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    print(f"  Test images : {len(test_dataset)}")

    model                   = load_model()
    y_true, y_pred, y_probs = get_predictions(model, test_loader)

    acc, auc, sens, spec    = print_report(y_true, y_pred, y_probs)
    plot_results(y_true, y_pred, y_probs, acc, auc, sens, spec)

    print("\n  ✅ Evaluation complete!")