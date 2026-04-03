"""
setup_5grade.py
APTOS 2019 — 5-Grade DR Classification Dataset Setup
Grade 0: No DR
Grade 1: Mild DR
Grade 2: Moderate DR
Grade 3: Severe DR
Grade 4: Proliferative DR
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TRAIN_CSV  = "aptos_data/train_1.csv"
VAL_CSV    = "aptos_data/valid.csv"
TRAIN_IMGS = "aptos_data/train_images"
VAL_IMGS   = "aptos_data/val_images"
OUTPUT_DIR = "dataset_5grade"
TEST_RATIO = 0.15
RANDOM_SEED= 42

GRADE_NAMES = {
    0: "Grade0_NoDR",
    1: "Grade1_Mild",
    2: "Grade2_Moderate",
    3: "Grade3_Severe",
    4: "Grade4_Proliferative"
}

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def setup():
    print("═" * 60)
    print("  APTOS 2019 — 5-Grade Dataset Setup")
    print("═" * 60)

    # Load CSVs
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    # Add folder name + filename
    for df in [train_df, val_df]:
        df["folder"]   = df["diagnosis"].map(GRADE_NAMES)
        df["filename"] = df["id_code"] + ".png"

    # Carve test from train (stratified)
    train_df, test_df = train_test_split(
        train_df,
        test_size    = TEST_RATIO,
        stratify     = train_df["diagnosis"],
        random_state = RANDOM_SEED
    )

    print(f"\n  {'Grade':<25} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print(f"  {'─'*52}")

    for grade, name in GRADE_NAMES.items():
        tr = (train_df["diagnosis"] == grade).sum()
        vl = (val_df["diagnosis"]   == grade).sum()
        te = (test_df["diagnosis"]  == grade).sum()
        print(f"  {name:<25} {tr:>6} {vl:>6} {te:>6} {tr+vl+te:>7}")

    print(f"  {'─'*52}")
    print(f"  {'TOTAL':<25} "
          f"{len(train_df):>6} "
          f"{len(val_df):>6} "
          f"{len(test_df):>6} "
          f"{len(train_df)+len(val_df)+len(test_df):>7}")

    # Create folders
    for split in ["train", "val", "test"]:
        for name in GRADE_NAMES.values():
            os.makedirs(
                os.path.join(OUTPUT_DIR, split, name),
                exist_ok=True
            )

    # Copy images
    print("\n── Copying images ──")

    def copy_split(df, img_dir, split_name):
        ok = miss = 0
        for _, row in df.iterrows():
            src = os.path.join(img_dir, row["filename"])
            dst = os.path.join(OUTPUT_DIR, split_name,
                               row["folder"], row["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dst)
                ok += 1
            else:
                miss += 1
        print(f"  {split_name:<8}: {ok} copied | {miss} missing")

    copy_split(train_df, TRAIN_IMGS, "train")
    copy_split(val_df,   VAL_IMGS,   "val")
    copy_split(test_df,  TRAIN_IMGS, "test")

    # Verify
    print("\n── Folder Verification ──")
    for split in ["train", "val", "test"]:
        for name in GRADE_NAMES.values():
            folder = os.path.join(OUTPUT_DIR, split, name)
            count  = len(os.listdir(folder))
            print(f"  {split}/{name}: {count}")

    # Overlap check
    print("\n── Overlap Check ──")
    train_files = set()
    test_files  = set()
    for name in GRADE_NAMES.values():
        tf = os.path.join(OUTPUT_DIR, "train", name)
        te = os.path.join(OUTPUT_DIR, "test",  name)
        if os.path.exists(tf):
            train_files.update(os.listdir(tf))
        if os.path.exists(te):
            test_files.update(os.listdir(te))

    overlap = train_files & test_files
    if not overlap:
        print("  ✅ No overlap between train and test!")
    else:
        print(f"  ❌ Overlap found: {len(overlap)} files")

    print(f"\n  ✅ Dataset ready → {OUTPUT_DIR}/")
    print("  Next: py train_5grade.py")
    print("═" * 60)


if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        print(f"  Deleting old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    setup()