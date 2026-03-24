import cv2
import numpy as np
from pathlib import Path

# ── HARDWARE CONFIG ──
IMG_SIZE    = 512
NUM_WORKERS = 6
PIN_MEMORY  = True
# ─────────────────────


def crop_black_borders(img: np.ndarray, tolerance: int = 7) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    mask = gray > tolerance
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return img[rmin:rmax+1, cmin:cmax+1]


def ben_graham_preprocess(img: np.ndarray, img_size: int = 512) -> np.ndarray:
    img = img.astype(np.uint8)

    radius = int(img_size / 30)
    if radius % 2 == 0:
        radius += 1

    blurred = cv2.GaussianBlur(img, (radius, radius), sigmaX=0)
    result  = cv2.addWeighted(img, 4, blurred, -4, 128)

    return result


def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: tuple  = (8, 8)) -> np.ndarray:
    lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def full_preprocess(image_path: str, img_size: int = 512) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black_borders(img)
    img = ben_graham_preprocess(img, img_size=img_size)
    img = apply_clahe(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LANCZOS4)

    return img


# ✅🔥 GLOBAL FUNCTION (FIXED)
def process_one(row_dict, output_dir_str, img_size):
    output_dir = Path(output_dir_str)

    src  = row_dict['image_path']
    lbl  = row_dict['label']
    name = row_dict['image_name']

    dst  = output_dir / str(lbl) / name

    if dst.exists():
        return str(dst)

    try:
        img = full_preprocess(src, img_size)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(dst)
    except Exception as e:
        return f"ERROR: {src} — {e}"


# ── MAIN FUNCTION ──
def preprocess_dataset(csv_path: str,
                       output_dir: str,
                       img_size: int = 512,
                       num_workers_os: int = 6):

    import pandas as pd
    from tqdm.contrib.concurrent import process_map
    from functools import partial

    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)

    for c in range(5):
        (output_dir / str(c)).mkdir(parents=True, exist_ok=True)

    rows = df.to_dict('records')

    print(f"Preprocessing {len(rows)} images → {output_dir}")
    print(f"Using {num_workers_os} parallel workers...")

    # ✅ partial use kiya (args pass karne ke liye)
    worker_fn = partial(process_one,
                        output_dir_str=str(output_dir),
                        img_size=img_size)

    results = process_map(
        worker_fn,
        rows,
        max_workers=num_workers_os,
        chunksize=50,
        desc="Preprocessing"
    )

    errors = [r for r in results if isinstance(r, str) and r.startswith("ERROR")]

    print(f"\n✓ Done. {len(rows)-len(errors)} succeeded, {len(errors)} failed.")

    if errors:
        for e in errors[:10]:
            print(f"  {e}")

    df['preprocessed_path'] = results
    updated_csv = Path(csv_path).parent / (Path(csv_path).stem + "_preprocessed.csv")
    df.to_csv(updated_csv, index=False)

    print(f"✓ Updated CSV saved → {updated_csv}")


if __name__ == "__main__":
    preprocess_dataset(
        csv_path   = "data/splits/full_dataset.csv",
        output_dir = "data/preprocessed",
        img_size   = 512,
        num_workers_os = 6
    )