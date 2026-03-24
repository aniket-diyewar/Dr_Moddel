import torch
import torch.nn as nn
from pathlib import Path


def freeze_backbones(model: nn.Module):
    """Stage 1: freeze all 3 backbones, train only fusion + classifier."""
    base = model.module if hasattr(model, 'module') else model
    for name in ['efficientnet', 'resnet', 'vit']:
        for param in getattr(base, name).parameters():
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 — Backbones FROZEN | Trainable params: {trainable:,}")


def unfreeze_backbones(model: nn.Module, last_n_blocks: int = 4):
    """
    Stage 2: unfreeze last N blocks of each backbone for end-to-end fine-tuning.
    last_n_blocks=4 is recommended for 12GB VRAM.
    """
    base = model.module if hasattr(model, 'module') else model

    # EfficientNet — unfreeze last N blocks
    blocks = list(base.efficientnet.blocks)
    for block in blocks[-last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    # Always unfreeze head
    for param in base.efficientnet.conv_head.parameters():
        param.requires_grad = True
    for param in base.efficientnet.bn2.parameters():
        param.requires_grad = True

    # ResNet — unfreeze layer4
    for param in base.resnet.layer4.parameters():
        param.requires_grad = True

    # ViT — unfreeze last N transformer blocks
    vit_blocks = list(base.vit.blocks)
    for block in vit_blocks[-last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in base.vit.norm.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Stage 2 — Last {last_n_blocks} blocks UNFROZEN | "
          f"Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")


def get_optimizer_stage1(model: nn.Module, lr: float = 1e-3):
    """Stage 1 optimizer: only fusion + classifier."""
    base   = model.module if hasattr(model, 'module') else model
    params = list(base.fusion.parameters()) + list(base.classifier.parameters())
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def get_optimizer_stage2(model: nn.Module,
                          backbone_lr: float = 1e-5,
                          head_lr    : float = 1e-4):
    """
    Stage 2 optimizer: differential learning rates.
    Backbones get 10× lower LR than fusion/classifier —
    they are already pretrained; we want gentle fine-tuning.
    """
    base = model.module if hasattr(model, 'module') else model
    return torch.optim.AdamW([
        {'params': base.efficientnet.parameters(), 'lr': backbone_lr},
        {'params': base.resnet.parameters(),       'lr': backbone_lr},
        {'params': base.vit.parameters(),          'lr': backbone_lr},
        {'params': base.fusion.parameters(),       'lr': head_lr},
        {'params': base.classifier.parameters(),   'lr': head_lr},
    ], weight_decay=1e-4)


def get_scheduler(optimizer, stage: int, epochs: int):
    """
    Stage 1: CosineAnnealingLR (fast convergence of head)
    Stage 2: CosineAnnealingWarmRestarts (careful backbone tuning)
    """
    if stage == 1:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7)


def save_checkpoint(model, optimizer, scheduler, epoch,
                    val_qwk, best_qwk, path):
    base = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch'       : epoch,
        'model_state' : base.state_dict(),
        'optim_state' : optimizer.state_dict(),
        'sched_state' : scheduler.state_dict(),
        'val_qwk'     : val_qwk,
        'best_qwk'    : best_qwk,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}  (QWK: {val_qwk:.4f})")


def load_checkpoint(model, path, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu')
    base = model.module if hasattr(model, 'module') else model
    base.load_state_dict(ckpt['model_state'])
    if optimizer:  optimizer.load_state_dict(ckpt['optim_state'])
    if scheduler:  scheduler.load_state_dict(ckpt['sched_state'])
    print(f"✓ Loaded checkpoint: epoch {ckpt['epoch']}, QWK {ckpt['val_qwk']:.4f}")
    return ckpt['epoch'], ckpt['best_qwk']


def print_model_summary(model, img_size: int = 512, batch_size: int = 2):
    """Print parameter count and estimated VRAM usage per batch size."""
    from torchinfo import summary
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    try:
        summary(model,
                input_size   = (batch_size, 3, img_size, img_size),
                col_names    = ["input_size","output_size","num_params","trainable"],
                device       = 'cpu',
                verbose      = 1)
    except Exception:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params    : {total:,}")
        print(f"Trainable params: {trainable:,}")
    print("="*60)

    # VRAM estimate
    param_mb = sum(p.numel() * p.element_size()
                   for p in model.parameters()) / 1024**2
    grad_mb  = param_mb            # gradients same size as params
    act_mb   = batch_size * 3 * img_size * img_size * 4 / 1024**2 * 50  # rough activations
    total_mb = param_mb + grad_mb + act_mb
    print(f"\nEstimated VRAM (batch={batch_size}, {img_size}×{img_size}):")
    print(f"  Parameters : {param_mb:.0f} MB")
    print(f"  Gradients  : {grad_mb:.0f} MB")
    print(f"  Activations: ~{act_mb:.0f} MB (rough estimate)")
    print(f"  Total      : ~{total_mb/1024:.1f} GB")
    print(f"\n  Recommended batch size for 12 GB VRAM: 16")