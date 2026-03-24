import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

LOG_PATH = "logs/training_history.csv"
OUT_PATH = "logs/training_curves.png"

def plot_history():
    df = pd.read_csv(LOG_PATH)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Training history — Diabetic Retinopathy Classifier",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.32)

    # Stage boundary
    s1_end = df[df['stage'] == 1]['epoch'].max()

    def add_stage_line(ax):
        ax.axvline(s1_end, color='#888', linestyle='--',
                   linewidth=0.8, label='Stage 1→2')

    epochs = df['epoch'].values

    # 1. Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, df['train_loss'], label='Train', color='#3B8BD4')
    ax1.plot(epochs, df['val_loss'],   label='Val',   color='#D85A30')
    add_stage_line(ax1)
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 2. QWK (primary metric)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, df['train_qwk'], label='Train', color='#3B8BD4')
    ax2.plot(epochs, df['val_qwk'],   label='Val',   color='#D85A30')
    ax2.axhline(0.85, color='#1D9E75', linestyle=':', linewidth=1.2,
                label='Target (0.85)')
    add_stage_line(ax2)
    ax2.set_title('Quadratic Weighted Kappa (primary)')
    ax2.set_xlabel('Epoch'); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # 3. AUC-ROC
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, df['val_auc'], label='Val AUC', color='#9F4AB0')
    ax3.axhline(0.95, color='#1D9E75', linestyle=':', linewidth=1.2,
                label='Target (0.95)')
    add_stage_line(ax3)
    ax3.set_title('AUC-ROC (macro OvR)')
    ax3.set_xlabel('Epoch'); ax3.set_ylim(0.5, 1)
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # 4. Referral sensitivity
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, df['val_ref_sens'], label='Referral sensitivity',
             color='#D85A30')
    ax4.axhline(0.90, color='#1D9E75', linestyle=':', linewidth=1.2,
                label='Target (0.90)')
    add_stage_line(ax4)
    ax4.set_title('Referral sensitivity (grades 2–4)')
    ax4.set_xlabel('Epoch'); ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # 5. QWK convergence speed (stage 2 only)
    s2 = df[df['stage'] == 2]
    ax5 = fig.add_subplot(gs[1, 1])
    if len(s2) > 1:
        ax5.plot(s2['epoch'], s2['val_qwk'], color='#D85A30', marker='o',
                 markersize=3)
        best_epoch = s2.loc[s2['val_qwk'].idxmax(), 'epoch']
        best_qwk   = s2['val_qwk'].max()
        ax5.axvline(best_epoch, color='#1D9E75', linestyle='--',
                    label=f'Best: {best_qwk:.4f}')
    ax5.set_title('Stage 2 QWK detail')
    ax5.set_xlabel('Epoch')
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # 6. Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    best = df.loc[df['val_qwk'].idxmax()]
    summary = (
        f"Best checkpoint\n"
        f"{'─'*28}\n"
        f"Epoch        : {best['epoch']:.0f}\n"
        f"Stage        : {best['stage']:.0f}\n"
        f"Val QWK      : {best['val_qwk']:.4f}\n"
        f"Val AUC      : {best['val_auc']:.4f}\n"
        f"Ref. sens.   : {best['val_ref_sens']:.4f}\n"
        f"{'─'*28}\n"
        f"Total epochs : {len(df)}\n"
        f"S1 epochs    : {len(df[df['stage']==1])}\n"
        f"S2 epochs    : {len(df[df['stage']==2])}\n"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f0', alpha=0.8))

    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Training curves saved → {OUT_PATH}")

if __name__ == "__main__":
    plot_history()
