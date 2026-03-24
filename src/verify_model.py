import torch
import time
from model      import build_model, DRClassifier, DEVICE, N_GPUS
from losses     import build_loss
from model_utils import (freeze_backbones, unfreeze_backbones,
                          get_optimizer_stage1, get_optimizer_stage2,
                          print_model_summary)


def run_all_checks():
    print("="*60)
    print(f"Phase 3 verification")
    print(f"Device  : {DEVICE}")
    print(f"GPUs    : {N_GPUS} × {torch.cuda.get_device_name(0) if N_GPUS > 0 else 'None'}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB"
          if N_GPUS > 0 else "")
    print("="*60)

    # ── 1. Build model ────────────────────────────────────
    print("\n[1/6] Building ensemble model...")
    model = build_model('ensemble', pretrained=True)
    print(f"  ✓ Model on {DEVICE}")

    # ── 2. Model summary ──────────────────────────────────
    print("\n[2/6] Model summary...")
    print_model_summary(model, img_size=512, batch_size=2)

    # ── 3. Forward pass ───────────────────────────────────
    print("\n[3/6] Forward pass test (batch=4, 512×512)...")
    dummy = torch.randn(4, 3, 512, 512).to(DEVICE)
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == (4, 5), f"Expected (4,5), got {logits.shape}"
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4).to(DEVICE), atol=1e-5)
    print(f"  ✓ Output shape : {logits.shape}")
    print(f"  ✓ Probs sum    : {probs.sum(dim=1).tolist()}")

    # ── 4. Loss function ──────────────────────────────────
    print("\n[4/6] Loss function test...")
    loss_fn = build_loss("data/splits/train.csv", DEVICE)
    labels  = torch.randint(0, 5, (4,)).to(DEVICE)
    loss, breakdown = loss_fn(logits.detach(), labels)
    assert loss.item() > 0, "Loss should be positive"
    print(f"  ✓ Total loss : {loss.item():.4f}")
    for k, v in breakdown.items():
        print(f"     {k:<10}: {v:.4f}")

    # ── 5. Backward pass ──────────────────────────────────
    print("\n[5/6] Backward pass + gradient test...")
    freeze_backbones(model)
    opt    = get_optimizer_stage1(model)
    logits = model(dummy)
    loss, _ = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    # Check gradients flow into fusion
    base = model.module if hasattr(model, 'module') else model
    fusion_grad = base.fusion.attention[0].weight.grad
    assert fusion_grad is not None, "No gradient in fusion layer!"
    assert not torch.isnan(fusion_grad).any(), "NaN gradient detected!"
    print(f"  ✓ Fusion layer gradient norm : {fusion_grad.norm():.4f}")

    # ── 6. Attention weights ──────────────────────────────
    print("\n[6/6] Attention weight test...")
    base  = model.module if hasattr(model, 'module') else model
    with torch.no_grad():
        weights = base.get_attention_weights(dummy)
    assert weights.shape == (4, 3)
    print(f"  ✓ Attention weights shape: {weights.shape}")
    print(f"  Sample attention (backbone weights per image):")
    for i, row in enumerate(weights[:2]):
        eff, res, vit = row.tolist()
        print(f"    Img {i}: EfficientNet={eff:.3f}  ResNet={res:.3f}  ViT={vit:.3f}")

    # ── 7. VRAM usage ─────────────────────────────────────
    if N_GPUS > 0:
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved  = torch.cuda.memory_reserved(0) / 1024**2
        print(f"\nVRAM after forward+backward:")
        print(f"  Allocated : {allocated:.0f} MB")
        print(f"  Reserved  : {reserved:.0f} MB")
        print(f"  Free      : {torch.cuda.get_device_properties(0).total_memory/1024**2 - reserved:.0f} MB")

    # ── 8. Throughput benchmark ───────────────────────────
    print("\n[Bonus] Throughput benchmark — 10 forward passes @ batch=16...")
    if N_GPUS > 0:
        batch = torch.randn(16, 3, 512, 512).to(DEVICE)
        # Warmup
        with torch.no_grad():
            for _ in range(2): model(batch)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(10): model(batch)
        torch.cuda.synchronize()
        elapsed      = time.time() - start
        imgs_per_sec = (10 * 16) / elapsed
        print(f"  ✓ {imgs_per_sec:.1f} images/sec")
        print(f"  ✓ {elapsed/10*1000:.1f} ms per batch")
        train_size   = 26852
        epoch_time   = train_size / imgs_per_sec / 60
        print(f"  ✓ Estimated training time/epoch: {epoch_time:.1f} min")

    print("\n" + "="*60)
    print("✓ ALL CHECKS PASSED — Phase 3 complete")
    print("  Ready for Phase 4 (training loop)")
    print("="*60)

    return model, loss_fn


if __name__ == "__main__":
    model, loss_fn = run_all_checks()
