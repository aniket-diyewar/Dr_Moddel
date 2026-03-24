import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from model          import build_model, DEVICE, N_GPUS
from losses         import build_loss
from dataset        import get_dataloaders
from metrics        import compute_metrics, print_metrics
from early_stopping import EarlyStopping
from model_utils    import (
    freeze_backbones, unfreeze_backbones,
    get_optimizer_stage1, get_optimizer_stage2,
    get_scheduler, save_checkpoint
)

CFG = {
    'train_csv'     : 'data/splits/train.csv',
    'val_csv'       : 'data/splits/val.csv',
    'img_size'      : 512,
    'batch_size'    : 16,
    'arch'          : 'ensemble',
    'num_classes'   : 5,
    'dropout'       : 0.4,
    's1_epochs'     : 5,
    's1_lr'         : 1e-3,
    's2_epochs'     : 40,
    's2_backbone_lr': 2e-6,   # ✅ FIX — reduced from 1e-5 (was causing overfitting)
    's2_head_lr'    : 2e-5,   # ✅ FIX — reduced from 1e-4
    's2_unfreeze_n' : 4,
    'grad_clip'     : 1.0,
    'weight_decay'  : 1e-4,
    'patience'      : 12,     # ✅ FIX — increased from 8 (more time to converge)
    'use_amp'       : torch.cuda.is_available(),
    'checkpoint_dir': 'checkpoints',
    'log_dir'       : 'logs',
    'project'       : 'diabetic-retinopathy-clinical',
    'run_name'      : 'ensemble-v1',
}


def train_one_epoch(model, loader, loss_fn, optimizer,
                    scaler, epoch, stage) -> dict:
    model.train()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    loss_breakdown = {k: 0.0 for k in ['ce', 'focal', 'smooth', 'ordinal']}

    pbar = tqdm(loader, desc=f"[S{stage}|E{epoch+1}] Train",
                leave=False, dynamic_ncols=True)

    for imgs, labels in pbar:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # ✅ FIX — removed duplicate forward pass that was running model twice per batch
        with autocast(device_type='cuda', enabled=CFG['use_amp']):
            logits          = model(imgs)
            loss, breakdown = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

        total_loss += loss.item()
        for k in loss_breakdown:
            loss_breakdown[k] += breakdown.get(k, 0.0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr'  : f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    n       = len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / n
    for k in loss_breakdown:
        metrics[f'loss_{k}'] = loss_breakdown[k] / n
    return metrics


@torch.no_grad()
def validate(model, loader, loss_fn, epoch, stage) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    epoch_label = epoch if isinstance(epoch, str) else epoch + 1
    pbar = tqdm(loader, desc=f"[S{stage}|E{epoch_label}] Val  ",
                leave=False, dynamic_ncols=True)

    for imgs, labels in pbar:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with autocast(device_type='cuda', enabled=CFG['use_amp']):
            logits  = model(imgs)
            loss, _ = loss_fn(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    metrics         = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(loader)
    return metrics


def run_stage(stage, model, loss_fn,
              train_loader, val_loader,
              early_stopper, history) -> tuple:

    epochs = CFG['s1_epochs'] if stage == 1 else CFG['s2_epochs']

    if stage == 1:
        optimizer = get_optimizer_stage1(model, lr=CFG['s1_lr'])
    else:
        optimizer = get_optimizer_stage2(
            model,
            backbone_lr = CFG['s2_backbone_lr'],
            head_lr     = CFG['s2_head_lr']
        )

    scheduler = get_scheduler(optimizer, stage, epochs)
    scaler    = GradScaler(device='cuda', enabled=CFG['use_amp'])

    print(f"\n{'━'*60}")
    print(f"  STAGE {stage} — {'Head warmup' if stage == 1 else 'Full fine-tune'}")
    print(f"  Epochs      : {epochs}")
    print(f"  AMP         : {CFG['use_amp']}")
    print(f"  Grad clip   : {CFG['grad_clip']}")
    if stage == 2:
        print(f"  Backbone LR : {CFG['s2_backbone_lr']}")
        print(f"  Head LR     : {CFG['s2_head_lr']}")
        print(f"  Patience    : {CFG['patience']}")
    print(f"{'━'*60}")

    for epoch in range(epochs):
        epoch_start = time.time()

        train_m = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, epoch, stage)
        val_m   = validate(model, val_loader, loss_fn, epoch, stage)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1:>3}/{epochs}  ({epoch_time:.0f}s)  Stage {stage}")
        print(f"  Train → loss: {train_m['loss']:.4f}  "
              f"QWK: {train_m['qwk']:.4f}  AUC: {train_m['auc']:.4f}")
        print(f"  Val   → loss: {val_m['loss']:.4f}  "
              f"QWK: {val_m['qwk']:.4f}  AUC: {val_m['auc']:.4f}  "
              f"RefSens: {val_m['ref_sensitivity']:.4f}")

        wandb.log({
            'epoch'              : epoch + (0 if stage == 1 else CFG['s1_epochs']),
            'stage'              : stage,
            'lr'                 : optimizer.param_groups[0]['lr'],
            'train/loss'         : train_m['loss'],
            'train/loss_ce'      : train_m.get('loss_ce', 0),
            'train/loss_focal'   : train_m.get('loss_focal', 0),
            'train/qwk'          : train_m['qwk'],
            'train/auc'          : train_m['auc'],
            'train/accuracy'     : train_m['accuracy'],
            'val/loss'           : val_m['loss'],
            'val/qwk'            : val_m['qwk'],
            'val/auc'            : val_m['auc'],
            'val/accuracy'       : val_m['accuracy'],
            'val/ref_sensitivity': val_m['ref_sensitivity'],
            'val/ref_specificity': val_m['ref_specificity'],
            **{f"val/sens_class{i}": val_m[f'sensitivity_c{i}'] for i in range(5)},
            'epoch_time_s'       : epoch_time,
        })

        history.append({
            'stage'       : stage,
            'epoch'       : epoch + 1,
            'train_loss'  : train_m['loss'],
            'train_qwk'   : train_m['qwk'],
            'val_loss'    : val_m['loss'],
            'val_qwk'     : val_m['qwk'],
            'val_auc'     : val_m['auc'],
            'val_ref_sens': val_m['ref_sensitivity'],
        })

        # Save every epoch — power-cut safe
        base = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch'      : epoch,
            'model_state': base.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'best_qwk'   : early_stopper.best_score,
        }, f"{CFG['checkpoint_dir']}/last_model.pt")

        stop = early_stopper(val_m['qwk'], model, optimizer, scheduler, epoch)
        if stop:
            break

    return model, history, early_stopper.should_stop


def _setup_common():
    Path(CFG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(CFG['log_dir']).mkdir(parents=True, exist_ok=True)

    print(f"\nDevice : {DEVICE}")
    print(f"GPUs   : {N_GPUS}")
    if N_GPUS > 0:
        for i in range(N_GPUS):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}  "
                  f"{props.total_memory/1024**3:.1f} GB VRAM")

    print("\nBuilding DataLoaders...")
    train_loader, val_loader = get_dataloaders(
        train_csv  = CFG['train_csv'],
        val_csv    = CFG['val_csv'],
        batch_size = CFG['batch_size'],
        img_size   = CFG['img_size'],
    )
    return train_loader, val_loader


def _finish_training(model, history, val_loader, loss_fn):
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(f"{CFG['log_dir']}/training_history.csv", index=False)
    print(f"Training history saved -> {CFG['log_dir']}/training_history.csv")

    print("\nFinal validation with best model...")
    final_m = validate(model, val_loader, loss_fn, epoch=0, stage='best')
    print_metrics(final_m, split='FINAL VAL')

    wandb.summary['best_val_qwk']         = final_m['qwk']
    wandb.summary['best_val_auc']         = final_m['auc']
    wandb.summary['best_ref_sensitivity'] = final_m['ref_sensitivity']
    wandb.summary['best_ref_specificity'] = final_m['ref_specificity']

    base = model.module if hasattr(model, 'module') else model
    torch.save(base.state_dict(), f"{CFG['checkpoint_dir']}/last_model.pt")

    wandb.finish()

    print(f"\n{'='*60}")
    print("Training complete")
    print(f"  Best checkpoint -> checkpoints/best_model.pt")
    print(f"  Ready for Phase 5 (clinical evaluation + Grad-CAM)")
    print(f"{'='*60}")
    return final_m


def train():
    wandb.init(
        project = CFG['project'],
        name    = CFG['run_name'],
        config  = CFG,
        tags    = ['ensemble', 'clinical', 'phase4']
    )

    train_loader, val_loader = _setup_common()

    print("\nBuilding model...")
    model   = build_model(CFG['arch'], pretrained=True)
    loss_fn = build_loss(CFG['train_csv'], DEVICE)

    # Stage 1
    early_stopper = EarlyStopping(
        patience        = CFG['patience'],
        checkpoint_path = f"{CFG['checkpoint_dir']}/best_model.pt"
    )
    freeze_backbones(model)
    model, history, _ = run_stage(
        stage=1, model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        early_stopper=early_stopper, history=[]
    )

    # Stage 2
    early_stopper = EarlyStopping(
        patience        = CFG['patience'],
        checkpoint_path = f"{CFG['checkpoint_dir']}/best_model.pt"
    )
    unfreeze_backbones(model, last_n_blocks=CFG['s2_unfreeze_n'])
    model, history, _ = run_stage(
        stage=2, model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        early_stopper=early_stopper, history=history
    )

    print("\nRestoring best weights...")
    early_stopper.load_best(model)
    final_m = _finish_training(model, history, val_loader, loss_fn)
    return model, history, final_m


def resume_stage2(checkpoint_path: str = "checkpoints/best_model.pt"):
    """
    Resumes Stage 2 from a saved checkpoint after power cut or crash.
    Run with: py -3.11 src/train.py --resume
    """
    wandb.init(
        project = CFG['project'],
        name    = CFG['run_name'] + '-resumed',
        config  = CFG,
        tags    = ['ensemble', 'clinical', 'phase4', 'resumed']
    )

    train_loader, val_loader = _setup_common()

    print("\nBuilding model (loading from checkpoint)...")
    model = build_model(CFG['arch'], pretrained=False)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        ckpt_path = Path(CFG['checkpoint_dir']) / 'last_model.pt'
        print(f"  best_model.pt not found — trying last_model.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {CFG['checkpoint_dir']}/\n"
            f"Cannot resume — start fresh with: py -3.11 src/train.py"
        )

    print(f"  Loading: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    base = model.module if hasattr(model, 'module') else model
    base.load_state_dict(ckpt['model_state'])

    saved_epoch = ckpt.get('epoch', 0)
    saved_qwk   = ckpt.get('best_qwk', 0.0)
    print(f"  Checkpoint epoch : {saved_epoch}")
    print(f"  Best QWK so far  : {saved_qwk:.4f}")

    loss_fn = build_loss(CFG['train_csv'], DEVICE)
    unfreeze_backbones(model, last_n_blocks=CFG['s2_unfreeze_n'])

    early_stopper = EarlyStopping(
        patience        = CFG['patience'],
        checkpoint_path = f"{CFG['checkpoint_dir']}/best_model.pt"
    )
    early_stopper.best_score = saved_qwk
    print(f"  EarlyStopping patience : {CFG['patience']} epochs from here")

    history = []
    model, history, _ = run_stage(
        stage        = 2,
        model        = model,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        early_stopper= early_stopper,
        history      = history
    )

    print("\nRestoring best weights...")
    early_stopper.load_best(model)
    final_m = _finish_training(model, history, val_loader, loss_fn)
    return model, history, final_m


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--resume':
        print("=" * 60)
        print("  RESUMING STAGE 2 FROM CHECKPOINT")
        print("=" * 60)
        model, history, metrics = resume_stage2()
    else:
        model, history, metrics = train()
