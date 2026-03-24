import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from preprocessing import full_preprocess, crop_black_borders, ben_graham_preprocess, apply_clahe

CSV_PATH = "data/splits/train.csv"
OUT_PATH = Path("data/splits/preprocessing_verification.png")

CLASS_NAMES = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative DR"}

def verify():
    df  = pd.read_csv(CSV_PATH)
    fig, axes = plt.subplots(5, 5, figsize=(22, 22))
    fig.suptitle("Preprocessing pipeline — one sample per grade\n"
                 "(Raw → Cropped → Ben Graham → CLAHE → Final resize)",
                 fontsize=14, fontweight='bold')

    step_titles = ["Raw", "Circle crop", "Ben Graham", "CLAHE", "Final (512×512)"]

    for class_id in range(5):
        sample = df[df['label'] == class_id].sample(1, random_state=7).iloc[0]
        path   = sample['image_path']

        # Step by step
        raw  = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        crop = crop_black_borders(raw)
        bg   = ben_graham_preprocess(crop, img_size=512)
        cl   = apply_clahe(bg)
        fin  = cv2.resize(cl, (512, 512))

        steps = [raw, crop, bg, cl, fin]

        for step_idx, (img, title) in enumerate(zip(steps, step_titles)):
            ax = axes[class_id][step_idx]
            ax.imshow(img)
            ax.axis('off')
            if class_id == 0:
                ax.set_title(title, fontsize=11, fontweight='bold')
            if step_idx == 0:
                ax.set_ylabel(f"Grade {class_id}\n{CLASS_NAMES[class_id]}",
                              fontsize=10, rotation=90)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✓ Verification image saved → {OUT_PATH}")

if __name__ == "__main__":
    verify()
