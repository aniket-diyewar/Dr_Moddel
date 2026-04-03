"""
setup_aptos.py
Converts APTOS CSVs into binary train/val/test folder structure
No_DR = grade 0  |  DR = grades 1,2,3,4
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TRAIN_CSV   = "aptos_data/train_1.csv"
VAL_CSV     = "aptos_data/valid.csv"
TRAIN_IMGS  = "aptos_data/train_images"
VAL_IMGS    = "aptos_data/val_images"
OUTPUT_DIR  = "dataset"
TEST_RATIO  = 0.15        # 15% of train → test
RANDOM_SEED = 42

# ─────────────────────────────────────────────

def to_binary(diagnosis):
    return "No_DR" if diagnosis == 0 else "DR"

def copy_images(df, img_dir, split):
    """Copy images from source folder into dataset/split/label/ folders."""
    ok, missing = 0, 0
    for _, row in df.iterrows():
        src = os.path.join(img_dir, row["filename"])
        dst = os.path.join(OUTPUT_DIR, split, row["label"], row["filename"])
        if os.path.exists(src):
            shutil.copy2(src, dst)
            ok += 1
        else:
            missing += 1
    return ok, missing

def setup():
    print("═" * 55)
    print("  APTOS 2019 — Binary Dataset Setup")
    print("═" * 55)

    # ── Load CSVs ──
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # ── Add binary label + filename ──
    for df in [train_df, val_df]:
        df["label"]    = df["diagnosis"].apply(to_binary)
        df["filename"] = df["id_code"] + ".png"

    # ── Carve test set from train (stratified) ──
    train_df, test_df = train_test_split(
        train_df,
        test_size   = TEST_RATIO,
        stratify    = train_df["label"],
        random_state= RANDOM_SEED
    )

    # ── Print summary ──
    print(f"\n  {'Split':<8} {'No_DR':>8} {'DR':>8} {'Total':>8}")
    print(f"  {'─'*36}")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        no_dr = (df["label"] == "No_DR").sum()
        dr    = (df["label"] == "DR").sum()
        print(f"  {name:<8} {no_dr:>8} {dr:>8} {len(df):>8}")

    # ── Create folders ──
    for split in ["train", "val", "test"]:
        for label in ["No_DR", "DR"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

    # ── Copy images ──
    print("\n── Copying images ──")

    ok, miss = copy_images(train_df, TRAIN_IMGS, "train")
    print(f"  Train : {ok} copied  |  {miss} missing")

    ok, miss = copy_images(val_df, VAL_IMGS, "val")
    print(f"  Val   : {ok} copied  |  {miss} missing")

    ok, miss = copy_images(test_df, TRAIN_IMGS, "test")
    print(f"  Test  : {ok} copied  |  {miss} missing")

    # ── Verify folder counts ──
    print("\n── Verifying folder counts ──")
    for split in ["train", "val", "test"]:
        for label in ["No_DR", "DR"]:
            folder = os.path.join(OUTPUT_DIR, split, label)
            count  = len(os.listdir(folder))
            print(f"  {split}/{label:<8} : {count} images")

    # ── Overlap check ──
    print("\n── Checking for overlap ──")
    train_files = set(os.listdir(os.path.join(OUTPUT_DIR, "train", "DR")) +
                      os.listdir(os.path.join(OUTPUT_DIR, "train", "No_DR")))
    test_files  = set(os.listdir(os.path.join(OUTPUT_DIR, "test",  "DR")) +
                      os.listdir(os.path.join(OUTPUT_DIR, "test",  "No_DR")))
    val_files   = set(os.listdir(os.path.join(OUTPUT_DIR, "val",   "DR")) +
                      os.listdir(os.path.join(OUTPUT_DIR, "val",   "No_DR")))

    tr_te = train_files & test_files
    tr_va = train_files & val_files
    te_va = test_files  & val_files

    if not tr_te and not tr_va and not te_va:
        print("  ✅ No overlap between any splits!")
    else:
        print(f"  ⚠️  train∩test  : {len(tr_te)}")
        print(f"  ⚠️  train∩val   : {len(tr_va)}")
        print(f"  ⚠️  test∩val    : {len(te_va)}")

    print(f"\n  ✅ Dataset ready → {OUTPUT_DIR}/")
    print("  Next: py train_dr.py")
    print("═" * 55)


if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        print(f"  ⚠️  '{OUTPUT_DIR}' already exists — deleting...")
        shutil.rmtree(OUTPUT_DIR)
    setup()