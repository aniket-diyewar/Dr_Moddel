"""
clahe_preprocess.py
CLAHE Preprocessing for APTOS 2019 Fundus Images
Enhances blood vessel & lesion visibility
"""

import cv2
import numpy as np
import os
import shutil
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

INPUT_DIR   = "dataset"           # original clean dataset
OUTPUT_DIR  = "dataset_clahe"     # CLAHE enhanced dataset
SPLITS      = ["train", "val", "test"]
CLASSES     = ["DR", "No_DR"]

# CLAHE parameters
CLIP_LIMIT    = 2.0    # contrast limit — higher = more enhancement
TILE_GRID     = (8, 8) # grid size — smaller = more local enhancement


# ─────────────────────────────────────────────
# CLAHE FUNCTION
# ─────────────────────────────────────────────

def apply_clahe(img_path):
    """
    Apply CLAHE to fundus image.
    
    Process:
    1. Read image
    2. Convert BGR → LAB color space
    3. Apply CLAHE only to L (luminance) channel
    4. Convert back LAB → BGR → RGB
    
    Why LAB? CLAHE on L channel only enhances
    contrast without distorting colors.
    """
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        # Try PIL if cv2 fails
        pil_img = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel only
    clahe = cv2.createCLAHE(
        clipLimit   = CLIP_LIMIT,
        tileGridSize = TILE_GRID
    )
    l_clahe = clahe.apply(l)
    
    # Merge back
    lab_clahe = cv2.merge([l_clahe, a, b])
    
    # Convert back to RGB
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    img_rgb   = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
    
    return img_rgb


# ─────────────────────────────────────────────
# PROCESS DATASET
# ─────────────────────────────────────────────

def process_dataset():
    print("═" * 55)
    print("  CLAHE Preprocessing — APTOS 2019")
    print("═" * 55)

    if os.path.exists(OUTPUT_DIR):
        print(f"\n  ⚠️  '{OUTPUT_DIR}' exists — deleting...")
        shutil.rmtree(OUTPUT_DIR)

    total_processed = 0
    total_failed    = 0

    for split in SPLITS:
        for cls in CLASSES:
            in_folder  = os.path.join(INPUT_DIR,  split, cls)
            out_folder = os.path.join(OUTPUT_DIR, split, cls)

            if not os.path.exists(in_folder):
                continue

            os.makedirs(out_folder, exist_ok=True)
            files = [f for f in os.listdir(in_folder)
                     if f.lower().endswith(('.png','.jpg','.jpeg'))]

            print(f"\n  Processing {split}/{cls} — {len(files)} images")

            for fname in tqdm(files, desc=f"  {split}/{cls}",
                              ncols=60, ascii=True):
                in_path  = os.path.join(in_folder,  fname)
                out_path = os.path.join(out_folder, fname)

                try:
                    img_clahe = apply_clahe(in_path)
                    # Save as PNG for lossless quality
                    Image.fromarray(img_clahe).save(out_path)
                    total_processed += 1
                except Exception as e:
                    print(f"\n  ❌ Failed: {fname} — {e}")
                    # Copy original if CLAHE fails
                    shutil.copy2(in_path, out_path)
                    total_failed += 1

    print(f"\n{'═'*55}")
    print(f"  ✅ Processed : {total_processed} images")
    print(f"  ❌ Failed    : {total_failed} images (originals copied)")
    print(f"  📁 Output    : {OUTPUT_DIR}/")
    print(f"{'═'*55}")

    # Verify counts
    print("\n── Folder Count Verification ──")
    for split in SPLITS:
        for cls in CLASSES:
            orig  = os.path.join(INPUT_DIR,  split, cls)
            clahe = os.path.join(OUTPUT_DIR, split, cls)
            if os.path.exists(orig) and os.path.exists(clahe):
                o_cnt = len(os.listdir(orig))
                c_cnt = len(os.listdir(clahe))
                match = "✅" if o_cnt == c_cnt else "❌ MISMATCH"
                print(f"  {match} {split}/{cls}: {o_cnt} → {c_cnt}")


# ─────────────────────────────────────────────
# VISUAL COMPARISON (saves sample grid)
# ─────────────────────────────────────────────

def save_comparison_grid():
    """
    Save a side-by-side comparison of
    original vs CLAHE enhanced images.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    print("\n── Generating comparison grid ──")

    fig = plt.figure(figsize=(16, 10),
                     facecolor='#03060f')
    fig.suptitle("CLAHE Enhancement — Original vs Processed",
                 fontsize=14, color='white',
                 fontfamily='monospace', y=0.98)

    samples = []
    for cls in CLASSES:
        folder = os.path.join(INPUT_DIR, "test", cls)
        if not os.path.exists(folder): continue
        files  = os.listdir(folder)[:3]
        for f in files:
            samples.append((cls, f))

    rows = len(samples)
    gs   = gridspec.GridSpec(rows, 2,
                              hspace=0.4, wspace=0.1)

    for i, (cls, fname) in enumerate(samples):
        orig_path  = os.path.join(INPUT_DIR,  "test", cls, fname)
        clahe_path = os.path.join(OUTPUT_DIR, "test", cls, fname)

        orig  = np.array(Image.open(orig_path).resize((224,224)))
        clahe = np.array(Image.open(clahe_path).resize((224,224)))

        # Original
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(orig)
        ax1.set_title(f"Original — {cls}",
                      color='#64748b', fontsize=8)
        ax1.axis('off')

        # CLAHE
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(clahe)
        ax2.set_title(f"CLAHE Enhanced — {cls}",
                      color='#00b4ff', fontsize=8)
        ax2.axis('off')

    plt.savefig("clahe_comparison.png",
                dpi=120, bbox_inches='tight',
                facecolor='#03060f')
    plt.close()
    print("  📊 Saved → clahe_comparison.png")
    print("  Open this file to visually confirm enhancement!")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Install tqdm if needed
    try:
        from tqdm import tqdm
    except ImportError:
        os.system("pip install tqdm")
        from tqdm import tqdm

    process_dataset()
    save_comparison_grid()

    print("\n  Next step: py train_clahe.py")