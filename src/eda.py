import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image

CSV_PATH = Path("data/splits/full_dataset.csv")
OUTPUT_DIR = Path("data/splits")

CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
CLASS_COLORS = ["#3B8BD4", "#1D9E75", "#EF9F27", "#D85A30", "#D4537E"]

def run_eda():
    df = pd.read_csv(CSV_PATH)

    # ── 1. Class Distribution ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Diabetic Retinopathy — Dataset EDA", fontsize=14, fontweight='bold')

    counts = df['label'].value_counts().sort_index()
    axes[0].bar([CLASS_NAMES[i] for i in counts.index], counts.values,
                color=CLASS_COLORS, edgecolor='white', linewidth=0.5)
    axes[0].set_title("Class distribution (absolute count)")
    axes[0].set_xlabel("DR Grade")
    axes[0].set_ylabel("Image count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f"{v:,}", ha='center', fontsize=9)

    # Imbalance ratio
    axes[1].pie(counts.values,
                labels=[CLASS_NAMES[i] for i in counts.index],
                colors=CLASS_COLORS,
                autopct='%1.1f%%',
                startangle=140,
                pctdistance=0.82)
    axes[1].set_title(f"Class proportion\n(Imbalance ratio: {counts.max()//counts.min()}:1)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_distribution.png", dpi=150)
    plt.show()
    print("✓ Distribution plot saved")

    # ── 2. Check image sizes ───────────────────────────────
    print("\nChecking image sizes (sample of 200 per class)...")
    size_records = []
    for class_id in range(5):
        class_df = df[df['label'] == class_id].sample(min(200, len(df[df['label']==class_id])),
                                                        random_state=42)
        for _, row in tqdm(class_df.iterrows(), total=len(class_df),
                           desc=f"  Class {class_id}", leave=False):
            try:
                with Image.open(row['image_path']) as img:
                    size_records.append({
                        'label' : class_id,
                        'width' : img.width,
                        'height': img.height,
                        'mode'  : img.mode
                    })
            except Exception as e:
                print(f"  ⚠ Corrupt image: {row['image_path']} — {e}")

    size_df = pd.DataFrame(size_records)
    print(f"\nImage size stats:")
    print(size_df[['width', 'height']].describe().round(1))
    print(f"\nColor modes: {size_df['mode'].value_counts().to_dict()}")

    # Warn if mixed sizes
    unique_sizes = size_df.groupby(['width', 'height']).size()
    if len(unique_sizes) > 1:
        print(f"⚠  Multiple image sizes found ({len(unique_sizes)} unique sizes)")
        print("   Resizing will be handled in preprocessing — this is fine.")
    else:
        w, h = unique_sizes.index[0]
        print(f"✓  All images are {w}×{h}")

    # ── 3. Sample images ───────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("One sample image per class", fontsize=13)
    for class_id in range(5):
        sample = df[df['label'] == class_id].sample(1, random_state=1).iloc[0]
        img = cv2.imread(sample['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[class_id].imshow(img)
        axes[class_id].set_title(f"Grade {class_id}\n{CLASS_NAMES[class_id]}", fontsize=10)
        axes[class_id].axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_samples.png", dpi=150)
    plt.show()
    print("✓ Sample images saved")

    # ── 4. Corruption scan ────────────────────────────────
    print("\nScanning ALL images for corruption...")
    corrupt = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Scanning"):
        try:
            img = cv2.imread(row['image_path'])
            if img is None:
                corrupt.append(row['image_path'])
        except:
            corrupt.append(row['image_path'])

    if corrupt:
        print(f"\n⚠  {len(corrupt)} corrupt images found:")
        for p in corrupt:
            print(f"   {p}")
        corrupt_df = pd.DataFrame({'corrupt_path': corrupt})
        corrupt_df.to_csv(OUTPUT_DIR / "corrupt_images.csv", index=False)
        print(f"   Saved to data/splits/corrupt_images.csv")
        print("   Remove or replace these before splitting.")
    else:
        print("✓ No corrupt images found — clean dataset!")

    return df

if __name__ == "__main__":
    run_eda()