"""
setup_combined.py
Combines APTOS 2019 + Messidor-2 + IDRiD
→ 5-Grade DR Dataset
"""

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG — updated for CSV
# ─────────────────────────────────────────────

# APTOS 2019
APTOS_CSV       = "aptos_data/train_1.csv"
APTOS_VAL_CSV   = "aptos_data/valid.csv"
APTOS_TRAIN_IMG = "aptos_data/train_images"
APTOS_VAL_IMG   = "aptos_data/val_images"

# Messidor-2
MESSIDOR_CSV    = "messidor2_data/messidor2.csv"
MESSIDOR_IMG    = "messidor2_data/images"

# IDRiD
IDRID_CSV       = "idrid_data/labels.csv"
IDRID_IMG       = "idrid_data/images"

OUTPUT_DIR      = "dataset_combined"
TEST_RATIO      = 0.15
RANDOM_SEED     = 42

GRADE_NAMES = {
    0: "Grade0_NoDR",
    1: "Grade1_Mild",
    2: "Grade2_Moderate",
    3: "Grade3_Severe",
    4: "Grade4_Proliferative"
}

# ─────────────────────────────────────────────
# LOAD APTOS
# ─────────────────────────────────────────────

def load_aptos():
    print("  Loading APTOS 2019...")
    train = pd.read_csv(APTOS_CSV)
    val   = pd.read_csv(APTOS_VAL_CSV)

    records = []

    # Train images
    for _, row in train.iterrows():
        fname = row["id_code"] + ".png"
        path  = os.path.join(APTOS_TRAIN_IMG, fname)
        if os.path.exists(path):
            records.append({
                "path"   : path,
                "grade"  : int(row["diagnosis"]),
                "source" : "aptos"
            })

    # Val images
    for _, row in val.iterrows():
        fname = row["id_code"] + ".png"
        path  = os.path.join(APTOS_VAL_IMG, fname)
        if os.path.exists(path):
            records.append({
                "path"   : path,
                "grade"  : int(row["diagnosis"]),
                "source" : "aptos_val"
            })

    df = pd.DataFrame(records)
    print(f"    APTOS: {len(df)} images loaded")
    print(f"    {df['grade'].value_counts().sort_index().to_dict()}")
    return df

# ─────────────────────────────────────────────
# LOAD MESSIDOR-2
# ─────────────────────────────────────────────

def load_messidor():
    if not os.path.exists(MESSIDOR_CSV):
        return pd.DataFrame()

    print("  Loading Messidor-2...")
    try:
        df_raw = pd.read_csv(MESSIDOR_CSV)
        
        # Hardcoded to your exact CSV columns
        img_col   = 'id_code'
        grade_col = 'diagnosis'

        records = []
        for _, row in df_raw.iterrows():
            fname = str(row[img_col]).strip()
            if not fname.endswith(('.jpg','.jpeg','.png')):
                fname += '.jpg'

            path  = os.path.join(MESSIDOR_IMG, fname)
            grade = int(row[grade_col])

            if os.path.exists(path) and grade in [0,1,2,3,4]:
                records.append({
                    "path"  : path,
                    "grade" : grade,
                    "source": "messidor2"
                })

        df = pd.DataFrame(records)
        print(f"    Messidor-2: {len(df)} images loaded")
        return df

    except Exception as e:
        print(f"    ❌ Messidor error: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# LOAD IDRiD
# ─────────────────────────────────────────────

def load_idrid():
    if not os.path.exists(IDRID_CSV):
        print("  ⚠️  IDRiD not found — skipping")
        return pd.DataFrame()

    print("  Loading IDRiD...")

    try:
        df_raw = pd.read_csv(IDRID_CSV)
        print(f"    Columns: {df_raw.columns.tolist()}")

        records = []
        for _, row in df_raw.iterrows():
            img_id = None
            grade  = None

            for col in df_raw.columns:
                col_l = col.lower()
                if any(x in col_l for x in ['image','file','id']):
                    img_id = str(row[col]).strip()
                if any(x in col_l for x in ['grade','retinopathy',
                                              'level','diagnosis']):
                    grade = int(row[col])

            if img_id is None or grade is None:
                continue

            for ext in ['.jpg', '.jpeg', '.png']:
                path = os.path.join(IDRID_IMG, img_id + ext)
                if os.path.exists(path):
                    if grade in [0, 1, 2, 3, 4]:
                        records.append({
                            "path"  : path,
                            "grade" : grade,
                            "source": "idrid"
                        })
                    break

        df = pd.DataFrame(records)
        print(f"    IDRiD: {len(df)} images loaded")
        if len(df) > 0:
            print(f"    {df['grade'].value_counts().sort_index().to_dict()}")
        return df

    except Exception as e:
        print(f"    ❌ IDRiD error: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# COMBINE + SPLIT
# ─────────────────────────────────────────────

def combine_and_split(dfs):
    combined = pd.concat(
        [df for df in dfs if len(df) > 0],
        ignore_index=True
    )
    combined = combined.dropna(subset=['path','grade'])
    combined['grade'] = combined['grade'].astype(int)

    print(f"\n  Total combined: {len(combined)} images")
    print("\n  Grade distribution (combined):")
    for g, name in GRADE_NAMES.items():
        count = (combined['grade'] == g).sum()
        bar   = '█' * (count // 30)
        print(f"    {name:<25} {count:>5}  {bar}")

    train_df, test_df = train_test_split(
        combined,
        test_size    = TEST_RATIO,
        stratify     = combined['grade'],
        random_state = RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size    = 0.12,
        stratify     = train_df['grade'],
        random_state = RANDOM_SEED
    )

    print(f"\n  {'Split':<8} {'Total':>6}", end="")
    for g in GRADE_NAMES:
        print(f"  G{g}:{'':>3}", end="")
    print()

    for name, df in [("Train", train_df),
                     ("Val",   val_df),
                     ("Test",  test_df)]:
        print(f"  {name:<8} {len(df):>6}", end="")
        for g in GRADE_NAMES:
            print(f"  {(df['grade']==g).sum():>5}", end="")
        print()

    return train_df, val_df, test_df

# ─────────────────────────────────────────────
# COPY FILES
# ─────────────────────────────────────────────
from tqdm import tqdm

def copy_split(df, split_name):
    ok = miss = err = 0
    print(f"\n  Copying → {split_name}/ ({len(df)} images)")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        grade_folder = GRADE_NAMES[row['grade']]
        out_dir = os.path.join(OUTPUT_DIR, split_name, grade_folder)
        os.makedirs(out_dir, exist_ok=True)

        src = row['path']
        if not os.path.exists(src):
            miss += 1
            continue

        fname  = f"{row['source']}_{os.path.basename(src)}"
        dst    = os.path.join(out_dir, fname)
        dst_png = dst.rsplit('.', 1)[0] + '.png'

        # Try PIL convert → .png  (cleanly closed each time)
        try:
            with Image.open(src) as img:
                img.convert("RGB").save(dst_png)
            ok += 1
            continue
        except Exception as e_pil:
            pass  # fall through to raw copy

        # Fallback: raw copy (keeps original extension)
        try:
            shutil.copy2(src, dst)
            ok += 1
        except Exception as e_copy:
            print(f"\n    ⚠️  Skipping {src}: {e_copy}")
            err += 1

    print(f"  {split_name:<8}: {ok} copied | {miss} missing | {err} errors")
    return ok


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  Combined Dataset Setup")
    print("  APTOS 2019 + Messidor-2 + IDRiD")
    print("═" * 60)

    if os.path.exists(OUTPUT_DIR):
        print(f"\n  Deleting old {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    print("\n── Loading Datasets ──")
    aptos    = load_aptos()
    messidor = load_messidor()
    idrid    = load_idrid()

    print("\n── Combining ──")
    train_df, val_df, test_df = combine_and_split(
        [aptos, messidor, idrid]
    )

    print("\n── Copying Files ──")
    copy_split(train_df, "train")
    copy_split(val_df,   "val")
    copy_split(test_df,  "test")

    print("\n── Final Counts ──")
    total_imgs = 0
    for split in ["train", "val", "test"]:
        for grade, name in GRADE_NAMES.items():
            folder = os.path.join(OUTPUT_DIR, split, name)
            if os.path.exists(folder):
                count = len(os.listdir(folder))
                total_imgs += count
                print(f"  {split}/{name}: {count}")

    print(f"\n  Total images: {total_imgs}")
    print(f"  ✅ Dataset ready → {OUTPUT_DIR}/")
    print("  Next: py train_combined.py")
    print("═" * 60)