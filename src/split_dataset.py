import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

CSV_PATH   = Path("data/splits/full_dataset.csv")
OUTPUT_DIR = Path("data/splits")

CLASS_NAMES = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative DR"}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

def stratified_split():
    df = pd.read_csv(CSV_PATH)

    # Remove corrupt images if scan was run
    corrupt_path = OUTPUT_DIR / "corrupt_images.csv"
    if corrupt_path.exists():
        corrupt = pd.read_csv(corrupt_path)['corrupt_path'].tolist()
        before  = len(df)
        df      = df[~df['image_path'].isin(corrupt)].reset_index(drop=True)
        print(f"Removed {before - len(df)} corrupt images")

    # ── Stratified split ──────────────────────────────────
    train_df, temp_df = train_test_split(
        df,
        test_size   = VAL_RATIO + TEST_RATIO,
        stratify    = df['label'],
        random_state= RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size   = TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify    = temp_df['label'],
        random_state= RANDOM_SEED
    )

    # Add split column
    train_df = train_df.copy(); train_df['split'] = 'train'
    val_df   = val_df.copy();   val_df['split']   = 'val'
    test_df  = test_df.copy();  test_df['split']  = 'test'

    # ── Verify stratification ─────────────────────────────
    print("\n" + "="*55)
    print(f"{'Split':<8} {'Total':>7}  " +
          "  ".join([f"C{i}({CLASS_NAMES[i][:4]})" for i in range(5)]))
    print("="*55)

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df['label'].value_counts().sort_index()
        row = f"{name:<8} {len(split_df):>7}  "
        row += "  ".join([f"{counts.get(i,0):>8}" for i in range(5)])
        print(row)

    print("="*55)
    print("\nClass % in each split (should be ~equal):")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        pcts = split_df['label'].value_counts(normalize=True).sort_index() * 100
        row = f"{name:<8}  " + "  ".join([f"{pcts.get(i,0):>6.1f}%" for i in range(5)])
        print(row)

    # ── Save ──────────────────────────────────────────────
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(  OUTPUT_DIR / "val.csv",   index=False)
    test_df.to_csv( OUTPUT_DIR / "test.csv",  index=False)

    # Also save combined with split column (useful for k-fold later)
    full = pd.concat([train_df, val_df, test_df])
    full.to_csv(OUTPUT_DIR / "full_dataset_with_splits.csv", index=False)

    print(f"\n✓ Saved: train.csv ({len(train_df)}), "
          f"val.csv ({len(val_df)}), test.csv ({len(test_df)})")
    print(f"  Location: {OUTPUT_DIR.resolve()}")

    return train_df, val_df, test_df

if __name__ == "__main__":
    stratified_split()