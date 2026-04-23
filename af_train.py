"""
Training script for the 2D Attraction Field model (Paper 1 adaptation).

Training follows the general approach from Paper 1:
  - Adam optimizer, starting lr 1e-4, decayed at 1/3 and 2/3 of total iters
  - Batch size 2 (matching Paper 1's batch_size=2)
  - Loss: L_field + L_cls + L_norm  (Paper 1, Eq. 4)

Validation computes the dense-curve Chamfer metric from Our Paper.
"""

import os
import csv
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from af_dataset import SnakeCurveDataset, collate_fn, build_splits
from af_model import AttractionFieldNet
from af_loss import total_loss
from af_inference import model_predict, predict_curve
from af_evaluate import evaluate_dataset


# ------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, Rf):
    model.train()
    total, lf_sum, lc_sum, ln_sum = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    for imgs, fields, closeness, dist_maps, _ in loader:
        imgs      = imgs.to(device)
        fields    = fields.to(device)
        closeness = closeness.to(device)
        dist_maps = dist_maps.to(device)

        pred_field, pred_logits = model(imgs)
        loss, Lf, Lc, Ln = total_loss(
            pred_field, pred_logits,
            fields, closeness, dist_maps,
            Rf=Rf,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total  += loss.item()
        lf_sum += Lf.item()
        lc_sum += Lc.item()
        ln_sum += Ln.item()
        n_batches += 1

    return {
        'loss':   total  / max(n_batches, 1),
        'Lfield': lf_sum / max(n_batches, 1),
        'Lcls':   lc_sum / max(n_batches, 1),
        'Lnorm':  ln_sum / max(n_batches, 1),
    }


# ------------------------------------------------------------------
@torch.no_grad()
def validate(model, val_indices, image_dir, label_dir, device, cfg):
    """
    Run inference on the validation set and compute mean Chamfer distance.
    Uses the full predict_curve pipeline (extract → NMS → Isomap).

    During training, validates on up to cfg['val_max_samples'] images
    (random subset) to keep wall-clock time reasonable.  Set to None
    or 0 to use the full validation set.
    """
    model.eval()

    val_max = cfg.get('val_max_samples', 100)
    rng = np.random.default_rng(cfg.get('seed', 42))
    if val_max and len(val_indices) > val_max:
        subset = rng.choice(len(val_indices), val_max, replace=False).tolist()
        used_indices = [val_indices[i] for i in subset]
    else:
        used_indices = val_indices

    val_ds = SnakeCurveDataset(
        image_dir, label_dir, used_indices,
        image_size=cfg['image_size'], Rc=cfg['Rc'], Rf=cfg['Rf'],
    )

    predictions = []
    gt_curves   = []

    for idx in range(len(val_ds)):
        img_t, _, _, _, gt_pts = val_ds[idx]
        img_t = img_t.unsqueeze(0).to(device)       # (1, 1, H, W)

        pred_field, pred_close = model_predict(model, img_t, device=device)

        curve_xyd = predict_curve(
            pred_field, pred_close,
            t_thresh=cfg['t_thresh'],
            Rf=cfg['Rf'],
            nms_radius=cfg['nms_radius'],
            n_neighbors=cfg['isomap_neighbors'],
        )

        predictions.append(curve_xyd)
        gt_curves.append(gt_pts)

    results = evaluate_dataset(
        predictions, gt_curves,
        image_size=cfg['image_size'],
        n_per_seg=cfg['n_per_seg'],
    )
    return results


# ------------------------------------------------------------------
def train(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    cfg: dict,
):
    """
    Main training entry point.

    cfg keys
    --------
    image_size    : int    (400)
    Rc            : float  closeness radius in pixels  (15.0)
    Rf            : float  field training radius       (8.0)
    t_thresh      : float  closeness threshold at inf  (0.5)
    nms_radius    : float  NMS suppression radius      (3.0)
    isomap_neighbors : int                             (15)
    n_per_seg     : int    Catmull-Rom density         (20)
    batch_size    : int                                (2)
    lr            : float                              (1e-4)
    n_epochs      : int                                (50)
    val_every     : int    validate every N epochs     (5)
    device        : str    'cuda' or 'cpu'
    base_ch       : int    UNet base channels          (32)
    head_res      : int    residual blocks per head    (6)
    val_frac      : float  fraction for validation     (0.2)
    num_workers   : int                                (4)
    seed          : int                                (42)
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    device = cfg.get('device', 'cpu')
    seed   = cfg.get('seed', 42)

    # ---- data splits -----------------------------------------------
    train_idx, val_idx = build_splits(
        n_total=4870,
        val_frac=cfg.get('val_frac', 0.2),
        seed=seed,
    )
    print(f'Train: {len(train_idx)} | Val: {len(val_idx)}')

    train_ds = SnakeCurveDataset(
        image_dir, label_dir, train_idx,
        image_size=cfg['image_size'], Rc=cfg['Rc'], Rf=cfg['Rf'],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get('batch_size', 2),
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu'),
    )

    # ---- model / optimizer -----------------------------------------
    model = AttractionFieldNet(
        in_ch=1,
        base_ch=cfg.get('base_ch', 32),
        head_res=cfg.get('head_res', 6),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,}')

    n_epochs   = cfg.get('n_epochs', 50)
    optimizer  = optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-4))

    # LR schedule: decay at 1/3 and 2/3 of training (mirroring Paper 1's policy)
    milestones = [n_epochs // 3, 2 * n_epochs // 3]
    scheduler  = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print(f'Val subset size for mid-training checks: {cfg.get("val_max_samples", 100)}')
    print(f'  (full validation in eval mode uses all {len(val_idx)} val images)')

    # ---- logging ---------------------------------------------------
    log_path = os.path.join(output_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'Lfield', 'Lcls', 'Lnorm',
                         'val_mean_chamfer', 'val_n_failed', 'elapsed_s'])

    best_chamfer = float('inf')
    val_every    = cfg.get('val_every', 5)
    t0           = time.time()

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, cfg['Rf'])
        scheduler.step()

        val_chamfer = float('nan')
        val_failed  = -1

        if epoch % val_every == 0 or epoch == n_epochs:
            print(f'  → running validation at epoch {epoch}…', flush=True)
            val_results = validate(model, val_idx, image_dir, label_dir, device, cfg)
            val_chamfer = val_results['mean_chamfer']
            val_failed  = val_results['n_failed']

            if val_chamfer < best_chamfer:
                best_chamfer = val_chamfer
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, 'best_model.pth'))

        elapsed = time.time() - t0
        print(
            f'Epoch {epoch:3d}/{n_epochs} | '
            f'loss={train_metrics["loss"]:.4f} '
            f'(Lf={train_metrics["Lfield"]:.4f} '
            f'Lc={train_metrics["Lcls"]:.4f} '
            f'Ln={train_metrics["Lnorm"]:.4f}) | '
            f'val_chamfer={val_chamfer:.4f} | '
            f'elapsed={elapsed:.0f}s',
            flush=True,
        )

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics['loss'], train_metrics['Lfield'],
                train_metrics['Lcls'], train_metrics['Lnorm'],
                val_chamfer, val_failed, round(elapsed, 1),
            ])

        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'model_epoch{epoch:04d}.pth'))

    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model_final.pth'))
    print(f'\nTraining complete. Best val Chamfer: {best_chamfer:.4f}')
    print(f'Checkpoints saved to: {ckpt_dir}')
    return model, val_idx


# ------------------------------------------------------------------
# Default config
# ------------------------------------------------------------------

DEFAULT_CFG = {
    # Data
    'image_size':       400,
    # Paper 1 hyperparameters (adapted to 2D pixel space)
    'Rc':               15.0,      # closeness radius (pixels)
    'Rf':               8.0,       # field training radius (pixels)
    't_thresh':         0.5,       # closeness threshold at inference
    'nms_radius':       3.0,       # NMS suppression radius (pixels)
    'isomap_neighbors': 15,        # Isomap n_neighbors
    # Evaluation
    'n_per_seg':        20,        # Catmull-Rom points per segment
    # Training
    'batch_size':       2,
    'lr':               1e-4,
    'n_epochs':         50,
    'val_every':        5,
    'val_frac':         0.2,
    'val_max_samples':  100,    # max images for mid-training Chamfer validation
    'num_workers':      4,
    'seed':             42,
    # Model
    'base_ch':          32,
    'head_res':         6,
    'device':           'cuda' if torch.cuda.is_available() else 'cpu',
}
