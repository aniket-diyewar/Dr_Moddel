import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

TRAIN_CSV = Path("data/splits/train.csv")

def get_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Layer 1: Class weights for loss function.
    Penalizes errors on minority classes more heavily.
    """
    labels = train_df['label'].values
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(5),
        y=labels
    )
    weight_tensor = torch.tensor(weights, dtype=torch.float)

    print("Class weights for loss function:")
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.4f}")

    return weight_tensor


def get_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Layer 2: Weighted random sampler.
    Oversamples minority classes at the batch level — every batch
    will have a roughly equal number of samples from all classes.
    """
    labels  = train_df['label'].values
    weights = compute_class_weight('balanced', classes=np.arange(5), y=labels)

    # Each sample's weight = weight of its class
    sample_weights = np.array([weights[lbl] for lbl in labels])
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True   # must be True for oversampling
    )
    print(f"\nWeightedRandomSampler ready — {len(sample_weights)} samples per epoch")
    return sampler


def verify_sampler(sampler: WeightedRandomSampler,
                   train_df: pd.DataFrame,
                   n_batches: int = 100,
                   batch_size: int = 32):
    """
    Layer 3 (verification): Simulate n_batches to confirm
    the sampler is producing balanced batches.
    """
    from torch.utils.data import DataLoader, TensorDataset

    labels  = torch.tensor(train_df['label'].values)
    dummy   = TensorDataset(torch.zeros(len(labels)), labels)
    loader  = DataLoader(dummy, batch_size=batch_size, sampler=sampler)

    class_counts = np.zeros(5, dtype=int)
    for i, (_, batch_labels) in enumerate(loader):
        if i >= n_batches:
            break
        for lbl in batch_labels.numpy():
            class_counts[lbl] += 1

    total = class_counts.sum()
    print(f"\nSampler verification ({n_batches} batches × {batch_size}):")
    print(f"  {'Class':<10} {'Count':>7}  {'%':>6}  {'Expected':>10}")
    print(f"  {'-'*40}")
    for i, count in enumerate(class_counts):
        print(f"  Class {i}    {count:>7}  {count/total*100:>5.1f}%  {'~20.0%':>10}")

    if max(class_counts)/min(class_counts) < 2.0:
        print("\n  ✓ Sampler is working — classes are balanced in batches")
    else:
        print("\n  ⚠ Sampler imbalance detected — check class weights")


if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_CSV)

    class_weights = get_class_weights(train_df)
    sampler       = get_sampler(train_df)
    verify_sampler(sampler, train_df)

    # Save weights for use in training
    torch.save(class_weights, "data/splits/class_weights.pt")
    print("\n✓ class_weights.pt saved to data/splits/")