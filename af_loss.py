"""
Loss function for the attraction field model (Paper 1, Eq. 1–4).

L = L_field + L_cls + L_norm                           (Eq. 4)

L_field  : L1 on attraction vectors, restricted to ||F_p||_2 <= Rf  (Eq. 1)
L_cls    : BCE on closeness map                                      (Eq. 2)
L_norm   : L1 on norm difference, restricted to ||F_p||_2 <= Rf     (Eq. 3)

Note: Paper 1 calls L_field "MSE loss" in the text but the equation (Eq.1)
shows L1 averaging (1/N * sum ||...||_1).  We follow the equation.
"""

import torch
import torch.nn.functional as F


def _rf_mask(dist_map: torch.Tensor, Rf: float) -> torch.Tensor:
    """Binary mask for pixels within Rf of the curve. dist_map: (B,1,H,W)."""
    return (dist_map <= Rf).float()                    # (B, 1, H, W)


# ------------------------------------------------------------------
def loss_field(
    pred_field: torch.Tensor,
    gt_field:   torch.Tensor,
    dist_map:   torch.Tensor,
    Rf: float,
) -> torch.Tensor:
    """
    L_field: mean L1 error on attraction vectors within Rf radius.
    pred_field : (B, 2, H, W)
    gt_field   : (B, 2, H, W)
    dist_map   : (B, 1, H, W)  ground-truth distance to curve
    """
    mask = _rf_mask(dist_map, Rf)                      # (B, 1, H, W)
    mask2 = mask.expand_as(pred_field)                 # (B, 2, H, W)
    err = (pred_field - gt_field).abs() * mask2
    n = mask2.sum().clamp(min=1.0)
    return err.sum() / n


def loss_cls(
    pred_logits:   torch.Tensor,
    gt_closeness:  torch.Tensor,
) -> torch.Tensor:
    """
    L_cls: binary cross-entropy on closeness map (Eq. 2).
    pred_logits  : (B, 1, H, W) raw logits
    gt_closeness : (B, 1, H, W) binary float {0, 1}
    """
    return F.binary_cross_entropy_with_logits(pred_logits, gt_closeness)


def loss_norm(
    pred_field: torch.Tensor,
    gt_field:   torch.Tensor,
    dist_map:   torch.Tensor,
    Rf: float,
) -> torch.Tensor:
    """
    L_norm: L1 on norm difference within Rf radius (Eq. 3).
    Regularises against the multiple-projections failure mode.
    """
    mask = _rf_mask(dist_map, Rf)                      # (B, 1, H, W)
    pred_norm = pred_field.norm(dim=1, keepdim=True)   # (B, 1, H, W)
    gt_norm   = gt_field.norm(dim=1, keepdim=True)     # (B, 1, H, W)
    err = (pred_norm - gt_norm).abs() * mask
    n = mask.sum().clamp(min=1.0)
    return err.sum() / n


# ------------------------------------------------------------------
def total_loss(
    pred_field:      torch.Tensor,
    pred_logits:     torch.Tensor,
    gt_field:        torch.Tensor,
    gt_closeness:    torch.Tensor,
    dist_map:        torch.Tensor,
    Rf: float,
):
    """
    Returns (total, Lfield, Lcls, Lnorm).
    All components are unweighted summed (L = Lfield + Lcls + Lnorm) as
    specified in Paper 1 Eq. 4.
    """
    Lf = loss_field(pred_field, gt_field, dist_map, Rf)
    Lc = loss_cls(pred_logits, gt_closeness)
    Ln = loss_norm(pred_field, gt_field, dist_map, Rf)
    return Lf + Lc + Ln, Lf, Lc, Ln
