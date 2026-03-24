import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────
# MINORITY CLASSES (1, 3, 4) — aggressive augmentation
# ─────────────────────────────────────────────────────────────
MINORITY_TRANSFORM = A.Compose([

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),

    A.Affine(
        translate_percent = {'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
        scale             = (0.90, 1.10),
        rotate            = (-30, 30),
         p                 = 0.6
    ),

    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),

    # ✅ FIX 2 — removed invalid alpha_affine argument
    A.ElasticTransform(
        alpha = 120,
        sigma = 6.0,
        p     = 0.2
    ),

    # ✅ FIX 3 — removed invalid shift_limit argument
    A.OpticalDistortion(distort_limit=0.2, p=0.2),

    A.RandomBrightnessContrast(
        brightness_limit = 0.2,
        contrast_limit   = 0.2,
        p                = 0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit = 10,
        sat_shift_limit = 20,
        val_shift_limit = 10,
        p               = 0.4
    ),
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),

    # ✅ FIX 4 — var_limit replaced with std_range
    A.GaussNoise(std_range=(0.02, 0.10), p=0.2),

    A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.4), p=0.2),
    A.MotionBlur(blur_limit=3, p=0.1),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),

    # ✅ FIX 5 — CoarseDropout updated to new API
    A.CoarseDropout(
        num_holes_range   = (1, 6),
        hole_height_range = (16, 32),
        hole_width_range  = (16, 32),
        fill              = 0,
        p                 = 0.3
    ),

    A.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────
# MAJORITY CLASS (0) + MODERATE (2) — conservative augmentation
# ─────────────────────────────────────────────────────────────
MAJORITY_TRANSFORM = A.Compose([

    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),

    A.Affine(
        translate_percent = {'x': (-0.03, 0.03), 'y': (-0.03, 0.03)},
        scale             = (0.95, 1.05),
        rotate            = (-15, 15),
       p                 = 0.4
    ),

    A.RandomBrightnessContrast(
        brightness_limit = 0.1,
        contrast_limit   = 0.1,
        p                = 0.3
    ),
    A.HueSaturationValue(
        hue_shift_limit = 5,
        sat_shift_limit = 10,
        val_shift_limit = 5,
        p               = 0.3
    ),

    # ✅ FIX 4 — var_limit replaced with std_range
    A.GaussNoise(std_range=(0.01, 0.05), p=0.15),

    A.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────
# VALIDATION / TEST — no augmentation, normalize only
# ─────────────────────────────────────────────────────────────
VAL_TRANSFORM = A.Compose([
    A.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────
# TEST-TIME AUGMENTATION (TTA) — 5 versions per image at inference
# ─────────────────────────────────────────────────────────────
TTA_TRANSFORMS = [
    # 1. Original
    A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    # 2. Horizontal flip
    A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    # 3. Vertical flip
    A.Compose([
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    # 4. Rotate 90
    A.Compose([
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
    # 5. Slight brightness shift
    A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]),
]


def get_transform(split: str, label: int = None):
    """
    Returns the correct transform based on split and class label.

    Args:
        split : 'train', 'val', or 'test'
        label : class label 0-4, only used during training

    Returns:
        albumentations Compose transform
    """
    if split in ('val', 'test'):
        return VAL_TRANSFORM

    # Training — class-aware selection
    if label in (1, 3, 4):       # minority classes → aggressive
        return MINORITY_TRANSFORM
    else:                         # class 0 (No DR) and class 2 (Moderate)
        return MAJORITY_TRANSFORM
