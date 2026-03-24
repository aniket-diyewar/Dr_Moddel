import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def compute_metrics(all_labels: list, all_preds: list,
                    all_probs: list) -> dict:
    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    # QWK
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')

    # AUC
    try:
        present_classes = np.unique(labels)
        if len(present_classes) < 2:
            auc = 0.0
        elif len(present_classes) < 5:
            y_bin        = label_binarize(labels, classes=list(range(5)))
            probs_subset = probs[:, present_classes].copy()
            row_sums     = probs_subset.sum(axis=1, keepdims=True)
            row_sums     = np.where(row_sums == 0, 1, row_sums)
            probs_subset = probs_subset / row_sums
            auc = roc_auc_score(
                y_bin[:, present_classes],
                probs_subset,
                multi_class='ovr',
                average='macro'
            )
        else:
            auc = roc_auc_score(
                labels, probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"  AUC error: {e}")
        auc = 0.0

    # Accuracy
    acc = (labels == preds).mean()

    # Per-class sensitivity
    cm          = confusion_matrix(labels, preds, labels=list(range(5)))
    row_sums    = cm.sum(axis=1)
    sensitivity = np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)

    # Referral metrics
    ref_true = (labels >= 2).astype(int)
    ref_pred = (preds  >= 2).astype(int)
    tp = ((ref_pred == 1) & (ref_true == 1)).sum()
    tn = ((ref_pred == 0) & (ref_true == 0)).sum()
    fp = ((ref_pred == 1) & (ref_true == 0)).sum()
    fn = ((ref_pred == 0) & (ref_true == 1)).sum()
    ref_sensitivity = tp / (tp + fn + 1e-8)
    ref_specificity = tn / (tn + fp + 1e-8)

    return {
        'qwk'            : round(float(qwk),  4),
        'auc'            : round(float(auc),  4),
        'accuracy'       : round(float(acc),  4),
        'ref_sensitivity': round(float(ref_sensitivity), 4),
        'ref_specificity': round(float(ref_specificity), 4),
        'sensitivity_c0' : round(float(sensitivity[0]), 4),
        'sensitivity_c1' : round(float(sensitivity[1]), 4),
        'sensitivity_c2' : round(float(sensitivity[2]), 4),
        'sensitivity_c3' : round(float(sensitivity[3]), 4),
        'sensitivity_c4' : round(float(sensitivity[4]), 4),
        'confusion_matrix': cm.tolist(),
    }


def print_metrics(metrics: dict, split: str = 'val'):
    cm = np.array(metrics['confusion_matrix'])
    print(f"\n{'='*55}")
    print(f"  {split.upper()} METRICS")
    print(f"{'='*55}")
    print(f"  QWK (primary)        : {metrics['qwk']:.4f}   target >= 0.85")
    print(f"  AUC-ROC (macro)      : {metrics['auc']:.4f}   target >= 0.95")
    print(f"  Accuracy             : {metrics['accuracy']:.4f}")
    print(f"  Referral sensitivity : {metrics['ref_sensitivity']:.4f}   target >= 0.90")
    print(f"  Referral specificity : {metrics['ref_specificity']:.4f}   target >= 0.85")
    print(f"\n  Per-class sensitivity:")
    for i, name in enumerate(CLASS_NAMES):
        bar = '|' * int(metrics[f'sensitivity_c{i}'] * 20)
        print(f"    Class {i} {name:<16}: "
              f"{metrics[f'sensitivity_c{i}']:.4f}  {bar}")
    print(f"\n  Confusion matrix:")
    print(f"  {'':>16}", '  '.join([f"P{i}" for i in range(5)]))
    for i, row in enumerate(cm):
        print(f"  True {CLASS_NAMES[i]:<12}", '  '.join([f"{v:3d}" for v in row]))
    print(f"{'='*55}")