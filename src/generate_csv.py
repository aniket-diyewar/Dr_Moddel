import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────
RAW_DIR    = Path("data/raw")
OUTPUT_DIR = Path("data/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}
# ──────────────────────────────────────────────────────────

def generate_csv():
    records = []

    for class_id in sorted([int(d.name) for d in RAW_DIR.iterdir() if d.is_dir()]):
        class_dir = RAW_DIR / str(class_id)
        images    = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.JPG"))

        print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {len(images)} images found")

        for img_path in tqdm(images, desc=f"  Class {class_id}", leave=False):
            records.append({
                "image_path" : str(img_path),
                "image_name" : img_path.name,
                "label"      : class_id,
                "class_name" : CLASS_NAMES[class_id]
            })

    df = pd.DataFrame(records)

    # ✅ Fix 1: Check duplicates
    duplicates = df[df.duplicated(subset="image_name", keep=False)]
    if len(duplicates) > 0:
        print(f"\n⚠ Found {len(duplicates)} duplicate image names!")

        # ✅ Fix 2: Make names unique using path
        df["image_name"] = df["label"].astype(str) + "_" + df.groupby("image_name").cumcount().astype(str) + "_" + df["image_name"]

        print("✅ Duplicate names fixed automatically!")

    # Sanity checks
    assert df["image_path"].apply(os.path.exists).all(), \
        "Some image paths do not exist — check your folder structure!"

    assert df["label"].nunique() == 5, \
        f"Expected 5 classes, found {df['label'].nunique()}"

    # ❌ Removed strict duplicate assert (handled above)

    output_path = OUTPUT_DIR / "full_dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✓ CSV saved → {output_path}")
    print(f"  Total images : {len(df)}")
    print(f"  Columns      : {list(df.columns)}")
    print(f"\nClass distribution:\n{df['label'].value_counts().sort_index()}")

    return df

if __name__ == "__main__":
    df = generate_csv()