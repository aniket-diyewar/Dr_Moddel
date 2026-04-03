"""
evaluate_combined.py
5-Grade DR — Test Set Evaluation

Outputs:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrix (saved as evaluate_combined.png)
  - Quadratic Weighted Kappa (official DR grading metric)
  - Overall accuracy + top-2 accuracy
  - Results saved to evaluate_combined_results.txt
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    cohen_kappa_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR  = "dataset_combined"
MODEL_PATH   = "best_model_combined.pth"
BATCH_SIZE   = 16
NUM_CLASSES  = 5
NUM_WORKERS  = 0
PIN_MEMORY   = False
OUTPUT_IMG   = "evaluate_combined.png"
OUTPUT_TXT   = "evaluate_combined_results.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Grade0_NoDR",
    "Grade1_Mild",
    "Grade2_Moderate",
    "Grade3_Severe",
    "Grade4_Proliferative"
]
SHORT_NAMES = ["G0\nNoDR", "G1\nMild", "G2\nMod.", "G3\nSevere", "G4\nProli."]

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model():
    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    feats = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(feats, 512),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    return m

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def run_inference(model, loader):
    model.eval()
    all_preds   = []
    all_labels  = []
    all_probs   = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            out  = model(imgs)
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(prob.cpu().tolist())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs)
    )

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(preds, labels, probs):
    acc      = (preds == labels).mean()
    qwk      = cohen_kappa_score(labels, preds, weights="quadratic")

    # Top-2 accuracy (pred within 1 grade of truth)
    top2_acc = (np.abs(preds - labels) <= 1).mean()

    # Per-class accuracy
    per_class_acc = {}
    for g in range(NUM_CLASSES):
        mask = labels == g
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[g]] = (preds[mask] == g).mean()

    # Classification report (precision, recall, F1)
    report = classification_report(
        labels, preds,
        target_names=[n.replace("Grade", "G") for n in CLASS_NAMES],
        digits=3
    )

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Per-class ROC-AUC (one-vs-rest)
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    try:
        roc_auc = roc_auc_score(labels_bin, probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = float("nan")

    return {
        "acc": acc,
        "qwk": qwk,
        "top2_acc": top2_acc,
        "per_class_acc": per_class_acc,
        "report": report,
        "cm": cm,
        "roc_auc": roc_auc
    }

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

def plot_results(metrics, split_name="Test"):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#0d1117")

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── Plot 1: Confusion Matrix ──
    ax = axes[0]
    cm = metrics["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm, ax=ax,
        annot=cm,          # show raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=SHORT_NAMES,
        yticklabels=SHORT_NAMES,
        linewidths=0.5,
        linecolor="#30363d",
        cbar=False,
        annot_kws={"size": 11, "color": "white", "weight": "bold"}
    )
    ax.set_title("Confusion Matrix\n(counts, color = row %)",
                 color="#e6edf3", fontsize=13, pad=10)
    ax.set_xlabel("Predicted", color="#c9d1d9")
    ax.set_ylabel("True Label", color="#c9d1d9")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # ── Plot 2: Per-Class Accuracy Bar ──
    ax = axes[1]
    classes = list(metrics["per_class_acc"].keys())
    accs    = [metrics["per_class_acc"][c] * 100 for c in classes]
    short   = [c.replace("Grade", "G").replace("_", "\n") for c in classes]

    colors = ["#238636" if a >= 70 else "#d29922" if a >= 50 else "#da3633"
              for a in accs]
    bars = ax.bar(short, accs, color=colors, edgecolor="#30363d", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom",
                color="#e6edf3", fontsize=10, fontweight="bold")

    ax.axhline(70, color="#238636", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(50, color="#d29922", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", color="#c9d1d9")
    ax.set_title(f"Per-Class Accuracy ({split_name})", color="#e6edf3", fontsize=13, pad=10)

    legend_patches = [
        mpatches.Patch(color="#238636", label="≥70% (good)"),
        mpatches.Patch(color="#d29922", label="50-70% (ok)"),
        mpatches.Patch(color="#da3633", label="<50% (bad)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=9)

    # ── Plot 3: Summary Metrics ──
    ax = axes[2]
    ax.axis("off")

    summary_text = [
        ("Overall Accuracy",    f"{metrics['acc']*100:.2f}%"),
        ("Top-2 Accuracy",      f"{metrics['top2_acc']*100:.2f}%"),
        ("Quadratic κ (QWK)",   f"{metrics['qwk']:.4f}"),
        ("Macro ROC-AUC",       f"{metrics['roc_auc']:.4f}"),
        ("", ""),
        ("— Per class —", ""),
    ]
    for cls, acc in metrics["per_class_acc"].items():
        short_name = cls.replace("Grade", "G").replace("_", " ")
        summary_text.append((short_name, f"{acc*100:.1f}%"))

    y = 0.95
    ax.text(0.5, y + 0.03, f"{split_name} Set Results",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="#e6edf3", ha="center")

    for label, val in summary_text:
        if label == "":
            y -= 0.04
            continue
        color = "#e6edf3" if val == "" else "#58a6ff"
        ax.text(0.05, y, label, transform=ax.transAxes,
                fontsize=11, color="#8b949e", va="top")
        if val:
            ax.text(0.95, y, val, transform=ax.transAxes,
                    fontsize=11, color=color, va="top", ha="right",
                    fontweight="bold")
        y -= 0.075

    # QWK interpretation
    qwk = metrics["qwk"]
    if qwk >= 0.81:
        interp, col = "Excellent (publication-ready)", "#3fb950"
    elif qwk >= 0.61:
        interp, col = "Good (near publication quality)", "#d29922"
    elif qwk >= 0.41:
        interp, col = "Moderate (needs improvement)", "#f85149"
    else:
        interp, col = "Fair (significant issues)", "#da3633"

    ax.text(0.5, 0.05, f"QWK: {interp}",
            transform=ax.transAxes, fontsize=10,
            color=col, ha="center", style="italic")

    plt.suptitle("5-Grade DR Classification — EfficientNet-B3",
                 color="#e6edf3", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"  Plot saved → {OUTPUT_IMG}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "═" * 56)
    print("  5-Grade DR Evaluation — EfficientNet-B3")
    print("═" * 56)

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Model not found: {MODEL_PATH}")
        print("     Run train_combined.py first.")
        return

    # Load model
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model   : EfficientNet-B3  ({total_params/1e6:.1f}M params)")
    print(f"  Weights : {MODEL_PATH}")
    print(f"  Device  : {DEVICE}")

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Evaluate on BOTH val and test sets
    results = {}
    for split in ["val", "test"]:
        split_dir = os.path.join(DATASET_DIR, split)
        if not os.path.exists(split_dir):
            print(f"  ⚠️  {split}/ not found — skipping")
            continue

        ds = ImageFolder(split_dir, transform=val_transform)
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )

        # Class count summary
        counts = [len(os.listdir(os.path.join(split_dir, c)))
                  for c in ds.classes]
        print(f"\n── {split.upper()} SET ({sum(counts)} images) ──")
        for nm, cnt in zip(ds.classes, counts):
            print(f"    {nm:<26} {cnt:>4}")

        print(f"\n  Running inference...")
        preds, labels, probs = run_inference(model, loader)
        metrics = compute_metrics(preds, labels, probs)
        results[split] = metrics

        print(f"\n  {'─'*48}")
        print(f"  Overall Accuracy  : {metrics['acc']*100:.2f}%")
        print(f"  Top-2 Accuracy    : {metrics['top2_acc']*100:.2f}%")
        print(f"  QWK (kappa)       : {metrics['qwk']:.4f}")
        print(f"  Macro ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"  {'─'*48}")

        print(f"\n  Per-Class Accuracy:")
        for cls, acc in metrics["per_class_acc"].items():
            s   = "✅" if acc >= 0.70 else "⚠️" if acc >= 0.50 else "❌"
            bar = "█" * int(acc * 20)
            print(f"    {s} {cls:<26} {bar:<20} {acc*100:.1f}%")

        print(f"\n  Classification Report:")
        print(metrics["report"])

    # Plot using test set (fallback to val)
    plot_split = "test" if "test" in results else "val"
    if plot_split in results:
        plot_results(results[plot_split], split_name=plot_split.title())

    # Save text results
    with open(OUTPUT_TXT, "w") as f:
        f.write("5-Grade DR — Evaluation Results\n")
        f.write("=" * 56 + "\n\n")
        for split, metrics in results.items():
            f.write(f"{'─'*48}\n")
            f.write(f"{split.upper()} SET\n")
            f.write(f"{'─'*48}\n")
            f.write(f"Overall Accuracy : {metrics['acc']*100:.2f}%\n")
            f.write(f"Top-2 Accuracy   : {metrics['top2_acc']*100:.2f}%\n")
            f.write(f"QWK              : {metrics['qwk']:.4f}\n")
            f.write(f"Macro ROC-AUC    : {metrics['roc_auc']:.4f}\n\n")
            f.write("Per-Class Accuracy:\n")
            for cls, acc in metrics["per_class_acc"].items():
                f.write(f"  {cls:<28} {acc*100:.1f}%\n")
            f.write(f"\nClassification Report:\n{metrics['report']}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(metrics["cm"]) + "\n\n")

    print(f"\n  Results saved → {OUTPUT_TXT}")

    # QWK guidance
    if "test" in results:
        qwk = results["test"]["qwk"]
    elif "val" in results:
        qwk = results["val"]["qwk"]
    else:
        return

    print(f"\n{'═'*56}")
    print(f"  QWK = {qwk:.4f}")
    if qwk >= 0.81:
        print("  Status: EXCELLENT — publication-ready quality")
    elif qwk >= 0.61:
        print("  Status: GOOD — close to publication quality")
        print("  Tip: Try unfreezing full backbone for 5-10 more epochs")
    elif qwk >= 0.41:
        print("  Status: MODERATE — needs more work")
        print("  Tip: Check which grade pair is most confused")
        print("       Consider class-specific threshold tuning")
    else:
        print("  Status: FAIR — major issues remain")
        print("  Tip: Delete ckpt_s2.pth and retrain Stage 2 with")
        print("       STAGE2_LR=5e-5 and BACKBONE_UNFREEZE=0.80")
    print("═" * 56)


if __name__ == "__main__":
    main()