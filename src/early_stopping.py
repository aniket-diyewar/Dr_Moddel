import numpy as np
import torch
from pathlib import Path


class EarlyStopping:
    """
    Monitors validation QWK. Stops training if no improvement
    for `patience` consecutive epochs. Saves best model automatically.
    """
    def __init__(self, patience: int = 8, min_delta: float = 1e-4,
                 checkpoint_path: str = "checkpoints/best_model.pt"):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_score  = -np.inf
        self.counter     = 0
        self.should_stop = False

    def __call__(self, val_qwk: float, model, optimizer,
                 scheduler, epoch: int) -> bool:
        """
        Returns True if training should stop.
        Saves checkpoint whenever a new best is found.
        """
        if val_qwk > self.best_score + self.min_delta:
            self.best_score = val_qwk
            self.counter    = 0
            self._save(model, optimizer, scheduler, epoch, val_qwk)
            print(f"  ✓ New best QWK: {val_qwk:.4f} — checkpoint saved")
            return False
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience} "
                  f"(best={self.best_score:.4f})")
            if self.counter >= self.patience:
                print(f"\n  Early stopping triggered after {epoch+1} epochs.")
                self.should_stop = True
                return True
            return False

    def _save(self, model, optimizer, scheduler, epoch, qwk):
        base = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch'       : epoch,
            'model_state' : base.state_dict(),
            'optim_state' : optimizer.state_dict(),
            'sched_state' : scheduler.state_dict(),
            'best_qwk'    : qwk,
        }, self.checkpoint_path)

    def load_best(self, model):
        """Restore best weights after training ends."""
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        base = model.module if hasattr(model, 'module') else model
        base.load_state_dict(ckpt['model_state'])
        print(f"  ✓ Best weights restored (epoch {ckpt['epoch']}, "
              f"QWK={ckpt['best_qwk']:.4f})")
        return ckpt