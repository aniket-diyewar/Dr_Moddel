"""
gradcam_dr.py
Grad-CAM Visualization for DR Detection
Shows WHICH part of fundus image the model looks at
"""

import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATASET_DIR  = "dataset"
MODEL_PATH   = "best_model_dr.pth"
NUM_CLASSES  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES  = ["DR", "No_DR"]
NUM_SAMPLES  = 8    # how many images to visualize

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model():
    model = efficientnet_b0(weights=None)
    num_features     = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH,
                          map_location=DEVICE, weights_only=True))
    print(f"  ✅ Model loaded from {MODEL_PATH}")
    return model.to(DEVICE).eval()


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────

class GradCAM:
    """
    Computes Grad-CAM heatmap for EfficientNet-B0.
    Hooks into the last conv layer of features block.
    """
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Hook into last conv block of EfficientNet features
        target_layer = model.features[-1]

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(img_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backprop for target class
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)

        # Resize to 224x224
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        # Normalize to 0-1
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx, output.softmax(dim=1)[0].detach().cpu().numpy()


# ─────────────────────────────────────────────
# DENORMALIZE IMAGE for display
# ─────────────────────────────────────────────

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


# ─────────────────────────────────────────────
# OVERLAY HEATMAP ON IMAGE
# ─────────────────────────────────────────────

def apply_heatmap(img_np, cam):
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap  = heatmap.astype(np.float32) / 255.0
    overlay  = 0.5 * img_np + 0.5 * heatmap
    return np.clip(overlay, 0, 1)


# ─────────────────────────────────────────────
# VISUALIZE SAMPLES
# ─────────────────────────────────────────────

def visualize_gradcam(model, dataset, num_samples=NUM_SAMPLES):
    gradcam = GradCAM(model)

    # Get equal samples from DR and No_DR
    dr_indices    = [i for i, (_, l) in enumerate(dataset.samples) if l == 0]
    nodr_indices  = [i for i, (_, l) in enumerate(dataset.samples) if l == 1]

    # Pick random samples from each class
    np.random.seed(42)
    dr_pick   = np.random.choice(dr_indices,   num_samples // 2, replace=False)
    nodr_pick = np.random.choice(nodr_indices, num_samples // 2, replace=False)
    indices   = list(dr_pick) + list(nodr_pick)

    fig, axes = plt.subplots(num_samples, 3,
                              figsize=(14, num_samples * 4))
    fig.suptitle("Grad-CAM — What the Model Sees in Fundus Images",
                 fontsize=16, fontweight='bold', y=1.01)

    col_titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    for row, idx in enumerate(indices):
        img_path, true_label = dataset.samples[idx]
        img_tensor = dataset[idx][0].unsqueeze(0).to(DEVICE)
        img_np     = denormalize(dataset[idx][0])

        # Generate Grad-CAM
        cam, pred_label, probs = gradcam.generate(img_tensor)
        overlay                = apply_heatmap(img_np, cam)

        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_label]
        conf      = probs[pred_label] * 100
        correct = "[OK]" if true_label == pred_label else "[WRONG]"


        # ── Col 1: Original ──
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(
            f"True: {true_name}\nPred: {pred_name} {correct}\nConf: {conf:.1f}%",
            fontsize=10, fontweight='bold',
            color='green' if true_label == pred_label else 'red'
        )
        axes[row, 0].axis('off')

        # ── Col 2: Heatmap only ──
        axes[row, 1].imshow(img_np)
        axes[row, 1].imshow(cam, cmap='jet', alpha=0.6)
        axes[row, 1].axis('off')

        # ── Col 3: Overlay ──
        axes[row, 2].imshow(overlay)
        axes[row, 2].axis('off')

    # Legend for heatmap colors
    legend = [
        mpatches.Patch(color='red',    label='High attention'),
        mpatches.Patch(color='yellow', label='Medium attention'),
        mpatches.Patch(color='blue',   label='Low attention'),
    ]
    fig.legend(handles=legend, loc='lower center',
               ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig("gradcam_results.png", dpi=120,
                bbox_inches='tight')
    plt.show()
    print("  📊 Saved → gradcam_results.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("\n══════════════════════════════════════")
    print("   Grad-CAM Visualization             ")
    print("══════════════════════════════════════")

    # Install opencv if needed
    try:
        import cv2
    except ImportError:
        print("  Installing opencv...")
        os.system("pip install opencv-python")
        import cv2

    test_dataset = ImageFolder(
        root=os.path.join(DATASET_DIR, "test"),
        transform=test_transform
    )
    print(f"  Test images : {len(test_dataset)}")

    model = load_model()

    print(f"\n  Generating Grad-CAM for {NUM_SAMPLES} samples "
          f"({NUM_SAMPLES//2} DR + {NUM_SAMPLES//2} No_DR)...")

    visualize_gradcam(model, test_dataset, num_samples=NUM_SAMPLES)

    print("\n  ✅ Grad-CAM complete!")
    print("  Open gradcam_results.png to see results!")