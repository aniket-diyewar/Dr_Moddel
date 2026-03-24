import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalDRLoss(nn.Module):
    """
    Three-component loss tuned for clinical DR grading.

    IMPORTANT: Class balance is handled by WeightedRandomSampler in the
    DataLoader. Therefore class_weights here are kept UNIFORM (all 1.0).
    Using both sampler + high class weights causes the model to collapse
    into predicting only minority classes for every input.
    """

    def __init__(self,
                 class_weights  : torch.Tensor,
                 focal_gamma    : float = 1.5,   # ✅ reduced from 2.0
                 focal_alpha    : float = 0.5,   # ✅ reduced from 0.75
                 label_smoothing: float = 0.1,
                 ordinal_weight : float = 0.1):  # ✅ reduced from 0.2
        super().__init__()
        self.class_weights   = class_weights
        self.focal_gamma     = focal_gamma
        self.focal_alpha     = focal_alpha
        self.label_smoothing = label_smoothing
        self.ordinal_weight  = ordinal_weight
        self.num_classes     = len(class_weights)

    def weighted_ce(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.class_weights)

    def focal_loss(self, logits, targets):
        ce    = F.cross_entropy(logits, targets,
                                weight=self.class_weights, reduction='none')
        pt    = torch.exp(-ce)
        focal = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce
        return focal.mean()

    def label_smoothing_loss(self, logits, targets):
        smooth    = self.label_smoothing / (self.num_classes - 1)
        soft      = torch.full_like(logits, smooth)
        soft.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft * log_probs).sum(dim=1).mean()

    def ordinal_penalty(self, logits, targets):
        """
        Penalizes grade-distance errors proportionally.
        Misclassifying Grade 3 as Grade 0 is penalized more than
        misclassifying Grade 3 as Grade 2.
        """
        preds     = logits.argmax(dim=1).float()
        targets_f = targets.float()
        dist      = torch.abs(preds - targets_f)
        return (dist * dist).mean() / (self.num_classes ** 2)

    def forward(self, logits, targets):
        l_ce      = self.weighted_ce(logits, targets)
        l_focal   = self.focal_loss(logits, targets)
        l_smooth  = self.label_smoothing_loss(logits, targets)
        l_ordinal = self.ordinal_penalty(logits, targets)

        total = (l_ce
                 + 0.5 * l_focal
                 + 0.3 * l_smooth
                 + self.ordinal_weight * l_ordinal)

        return total, {
            'ce'     : l_ce.item(),
            'focal'  : l_focal.item(),
            'smooth' : l_smooth.item(),
            'ordinal': l_ordinal.item(),
            'total'  : total.item()
        }


def build_loss(train_csv: str, device) -> ClinicalDRLoss:
    """
    Builds loss with UNIFORM class weights.

    WeightedRandomSampler in the DataLoader already ensures each batch
    has balanced class representation. Adding high class weights on top
    causes double-penalization of minority classes → model collapses to
    predicting only minority classes → Val QWK crashes to near 0.

    Solution: sampler handles balance, loss weights stay at 1.0.
    """
    # ✅ FIXED — uniform weights, sampler handles class balance
    class_weights = torch.ones(5, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.tolist()}  "
          f"(uniform — WeightedRandomSampler handles class balance)")

    return ClinicalDRLoss(class_weights=class_weights)