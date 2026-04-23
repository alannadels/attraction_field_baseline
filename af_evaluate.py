"""
Evaluation metric matching Our Paper's dense-curve Chamfer protocol
(Our_Paper.txt, Sections 5.1 and 6.5).

Protocol
--------
Both predicted and target curves are densified before comparison:
  1. Order control points by dist (predicted) / arclength (ground truth).
  2. Fit a dense Catmull-Rom spline in XY.
  3. Linearly interpolate dist along arclength of the dense curve.
  4. Compare in 3-D space [x, y, d] using symmetric Chamfer with p=1 (L1).

All spatial coordinates are normalised to [0, 1] by dividing by image_size
so that x, y, and d are on comparable scales.
"""

import numpy as np


# ------------------------------------------------------------------
# Catmull-Rom spline
# ------------------------------------------------------------------

def _catmull_rom_segment(p0, p1, p2, p3, n: int):
    """Catmull-Rom interpolation between p1 and p2 (n sample points)."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]   # (n, 1)
    t2, t3 = t ** 2, t ** 3
    # Standard Catmull-Rom formula
    q = 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )
    return q                                                  # (n, D)


def catmull_rom(ctrl_pts: np.ndarray, n_per_seg: int = 20) -> np.ndarray:
    """
    Dense Catmull-Rom spline through ordered control points.

    ctrl_pts : (N, D)
    Returns  : (M, D)  dense curve
    """
    N = len(ctrl_pts)
    if N < 2:
        return ctrl_pts.copy()
    if N == 2:
        t = np.linspace(0.0, 1.0, n_per_seg)[:, None]
        return (1 - t) * ctrl_pts[0] + t * ctrl_pts[1]

    # Phantom endpoints for natural boundary
    p_start = 2 * ctrl_pts[0] - ctrl_pts[1]
    p_end   = 2 * ctrl_pts[-1] - ctrl_pts[-2]
    pts     = np.vstack([p_start, ctrl_pts, p_end])           # (N+2, D)

    segments = []
    for i in range(1, len(pts) - 2):
        seg = _catmull_rom_segment(pts[i - 1], pts[i], pts[i + 1], pts[i + 2], n_per_seg)
        segments.append(seg)
    segments.append(ctrl_pts[-1:])                            # add final endpoint
    return np.vstack(segments)                                 # (M, D)


# ------------------------------------------------------------------
# Arclength helpers
# ------------------------------------------------------------------

def cumulative_arclength(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-length starting from 0. pts: (N, D)."""
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def arclength_normalised(pts: np.ndarray) -> np.ndarray:
    """Arc-length parameterisation in [0, 1]. pts: (N, D)."""
    arc = cumulative_arclength(pts)
    L = arc[-1]
    return arc / L if L > 0 else arc


# ------------------------------------------------------------------
# Densify a sparse curve to a dense [x, y, d] cloud
# ------------------------------------------------------------------

def densify_curve(
    ctrl_xy:  np.ndarray,    # (N, 2) ordered control points (normalised [0,1])
    ctrl_d:   np.ndarray,    # (N,)   dist values for each control point
    n_per_seg: int = 20,
) -> np.ndarray:
    """
    Produce a dense [x, y, d] cloud following Our Paper's protocol:
      1. Catmull-Rom interpolation of XY.
      2. Linear interpolation of d along arclength.

    Returns (M, 3) array.
    """
    # 1. Dense XY
    dense_xy = catmull_rom(ctrl_xy, n_per_seg)                 # (M, 2)

    # 2. Dense arclength parameterisation of the dense curve
    dense_arc = arclength_normalised(dense_xy)                 # (M,) in [0,1]

    # 3. Arclength parameterisation of sparse control points (for interp reference)
    sparse_arc = arclength_normalised(ctrl_xy)                 # (N,) in [0,1]

    # 4. Interpolate d linearly along arclength
    dense_d = np.interp(dense_arc, sparse_arc, ctrl_d)        # (M,)

    return np.concatenate([dense_xy, dense_d[:, None]], axis=1)   # (M, 3)


# ------------------------------------------------------------------
# Chamfer distance (symmetric, L1, p=1)
# ------------------------------------------------------------------

def chamfer_l1(A: np.ndarray, B: np.ndarray) -> float:
    """
    Symmetric Chamfer distance with L1 norm between clouds A and B.

    C(A, B) = 0.5 * (mean_a min_b ||a-b||_1  +  mean_b min_a ||b-a||_1)

    A : (M, D)
    B : (N, D)
    """
    # Pairwise L1:  (M, N)
    diff = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)  # (M, N)
    a_to_b = diff.min(axis=1).mean()
    b_to_a = diff.min(axis=0).mean()
    return 0.5 * (a_to_b + b_to_a)


# ------------------------------------------------------------------
# Single-sample evaluation
# ------------------------------------------------------------------

def evaluate_one(
    pred_xyd:     np.ndarray,   # (N, 3)  [x, y, d]  pixel coords + dist in [0,1]
    gt_curve_pts: np.ndarray,   # (M, 2)  [x, y]  pixel coords, already ordered
    image_size:   int   = 400,
    n_per_seg:    int   = 20,
) -> float:
    """
    Compute dense-curve Chamfer distance for one image.

    pred_xyd and gt_curve_pts are both in pixel coords [0, image_size].
    Returns float (lower is better); returns inf if prediction is degenerate.
    """
    if len(pred_xyd) < 2:
        return float('inf')

    # ---- predicted curve ----------------------------------------------
    # Sort by predicted dist
    order = np.argsort(pred_xyd[:, 2])
    pred_xyd = pred_xyd[order]

    pred_xy_norm = pred_xyd[:, :2] / image_size     # [0, 1]
    pred_d       = pred_xyd[:, 2]                   # already [0, 1]

    pred_dense = densify_curve(pred_xy_norm, pred_d, n_per_seg)   # (M, 3)

    # ---- ground-truth curve -------------------------------------------
    # gt_curve_pts is already ordered; derive dist from arclength
    gt_xy_norm = gt_curve_pts / image_size           # [0, 1]
    gt_d       = arclength_normalised(gt_xy_norm)    # [0, 1] by arclength

    gt_dense = densify_curve(gt_xy_norm, gt_d, n_per_seg)         # (K, 3)

    # ---- Chamfer -------------------------------------------------------
    return chamfer_l1(pred_dense, gt_dense)


# ------------------------------------------------------------------
# Dataset-level evaluation
# ------------------------------------------------------------------

def evaluate_dataset(
    predictions:   list,    # list of (N_i, 3) arrays [x, y, d] or (0,3)
    gt_curves:     list,    # list of (M_i, 2) arrays [x, y]
    image_size:    int = 400,
    n_per_seg:     int = 20,
) -> dict:
    """
    Evaluate a full set of predictions.

    Returns dict with keys:
      mean_chamfer : float  (primary metric, lower is better)
      per_sample   : list of float
      n_failed     : int    (samples where prediction was empty)
    """
    scores = []
    n_failed = 0
    for pred, gt in zip(predictions, gt_curves):
        s = evaluate_one(pred, gt, image_size=image_size, n_per_seg=n_per_seg)
        scores.append(s)
        if s == float('inf'):
            n_failed += 1

    finite = [s for s in scores if s != float('inf')]
    mean_ch = float(np.mean(finite)) if finite else float('inf')

    return {
        'mean_chamfer': mean_ch,
        'per_sample':   scores,
        'n_failed':     n_failed,
    }
