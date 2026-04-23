"""
Inference pipeline for the attraction field model (Paper 1, Section 2.2).

Steps (Eq. 5 and surrounding text):
  1. Forward pass → predicted field F_hat and closeness logits → sigmoid → C_hat
  2. Filter pixels: C_hat_p >= t  AND  ||F_hat_p||_2 <= Rf
  3. Shift pixel coords by F_hat to get point cloud on the curve
  4. Non-maximum suppression (NMS) with confidence = -||F_hat_p||_2
  5. Order unstructured cloud → 1-D Isomap embedding → dist in [0, 1]

Output: (N, 3) array  [x, y, dist]  in pixel coordinates.
"""

import numpy as np
from sklearn.manifold import Isomap
import torch


# ------------------------------------------------------------------
# 1.  Point-cloud extraction
# ------------------------------------------------------------------

def extract_point_cloud(
    pred_field:     np.ndarray,   # (H, W, 2)  attraction vectors
    pred_closeness: np.ndarray,   # (H, W)     closeness probability (post-sigmoid)
    t: float  = 0.5,
    Rf: float = 8.0,
):
    """
    Apply Eq. 5:  {F_hat_p + p | C_hat_p >= t,  ||F_hat_p||_2 <= Rf}

    Returns
    -------
    projected : (N, 2)  projected (x, y) pixel coords on the curve
    norms     : (N,)    ||F_hat_p||_2 for each kept pixel (used as NMS confidence)
    """
    H, W = pred_closeness.shape
    ys, xs = np.mgrid[0:H, 0:W]           # (H, W)

    norms = np.linalg.norm(pred_field, axis=2)   # (H, W)
    mask = (pred_closeness >= t) & (norms <= Rf)

    px = xs[mask].astype(np.float64)      # x = col
    py = ys[mask].astype(np.float64)      # y = row
    pixel_xy = np.stack([px, py], axis=1)            # (N, 2)

    field_at_mask = pred_field[mask]                 # (N, 2)  (dx, dy)
    projected = pixel_xy + field_at_mask             # (N, 2)

    return projected, norms[mask]


# ------------------------------------------------------------------
# 2.  Non-maximum suppression
# ------------------------------------------------------------------

def nms(
    points:      np.ndarray,   # (N, 2)
    norms:       np.ndarray,   # (N,)  ||F_hat_p||_2
    radius: float = 3.0,
) -> np.ndarray:
    """
    Greedy NMS: confidence = -||F_hat||_2  (smaller norm = more confident).
    Keep the highest-confidence point in each radius-neighbourhood.

    Returns filtered (M, 2) point array.
    """
    if len(points) == 0:
        return points

    # Sort ascending by norm (= descending confidence)
    order = np.argsort(norms)
    pts = points[order]

    kept_mask = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):
        if not kept_mask[i]:
            continue
        # Suppress all later points within radius
        diffs = pts[i + 1:] - pts[i]
        dists = np.linalg.norm(diffs, axis=1)
        close = dists < radius
        kept_mask[i + 1:][close] = False

    return pts[kept_mask]


# ------------------------------------------------------------------
# 3.  Isomap ordering (1-D embedding → dist parameter)
# ------------------------------------------------------------------

def order_with_isomap(
    points:      np.ndarray,   # (N, 2)  unordered point cloud
    n_neighbors: int = 15,
    max_cloud_size: int = 2000,
) -> tuple:
    """
    Reduce the point cloud to 1-D using Isomap, giving an intrinsic
    parameterisation of the curve (= dist).

    max_cloud_size: if the cloud exceeds this, randomly subsample before
    Isomap (avoids memory/time issues with untrained models).  After
    training the cloud is typically <500 points so this cap is never hit.

    Returns
    -------
    ordered_pts : (N, 2)  sorted by embedding value
    dist        : (N,)    normalised embedding in [0, 1]
    """
    n = len(points)
    if n == 0:
        return points, np.array([])
    if n < 3:
        t = np.linspace(0.0, 1.0, n)
        return points, t

    if n > max_cloud_size:
        idx = np.random.choice(n, max_cloud_size, replace=False)
        points = points[idx]
        n = max_cloud_size

    k = min(n_neighbors, n - 1)
    iso = Isomap(n_components=1, n_neighbors=k)
    emb = iso.fit_transform(points).ravel()          # (N,)

    e_min, e_max = emb.min(), emb.max()
    if e_max > e_min:
        dist = (emb - e_min) / (e_max - e_min)
    else:
        dist = np.zeros(n)

    order = np.argsort(dist)
    return points[order], dist[order]


# ------------------------------------------------------------------
# 4.  Full pipeline
# ------------------------------------------------------------------

def predict_curve(
    pred_field:     np.ndarray,   # (H, W, 2)
    pred_closeness: np.ndarray,   # (H, W)  sigmoid probabilities
    t_thresh: float = 0.5,
    Rf: float       = 8.0,
    nms_radius: float = 3.0,
    n_neighbors: int  = 15,
    max_cloud_size: int = 2000,
) -> np.ndarray:
    """
    Full inference pipeline returning an ordered curve in pixel space.

    Returns
    -------
    curve_xyd : (N, 3)  columns [x, y, dist]   dist in [0, 1]
                Returns empty (0, 3) array on failure.
    """
    # Step 1 – point cloud
    cloud, norms = extract_point_cloud(pred_field, pred_closeness, t=t_thresh, Rf=Rf)

    if len(cloud) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Step 2 – NMS
    cloud = nms(cloud, norms, radius=nms_radius)

    if len(cloud) < 2:
        return np.zeros((0, 3), dtype=np.float32)

    # Step 3 – Isomap ordering
    ordered, dist = order_with_isomap(cloud, n_neighbors=n_neighbors,
                                      max_cloud_size=max_cloud_size)

    return np.concatenate(
        [ordered.astype(np.float32), dist[:, None].astype(np.float32)],
        axis=1,
    )                                                  # (N, 3)


# ------------------------------------------------------------------
# 5.  Batch prediction helper (model → numpy)
# ------------------------------------------------------------------

def model_predict(model, img_tensor: torch.Tensor, device: str = 'cpu'):
    """
    Run one image through the model and return numpy arrays.

    img_tensor : (1, 1, H, W) torch tensor (batched, single image)
    Returns
    -------
    pred_field     : (H, W, 2)  numpy float32
    pred_closeness : (H, W)     numpy float32  (sigmoid applied)
    """
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        raw_field, raw_logits = model(img_tensor)       # (1,2,H,W), (1,1,H,W)

    pred_field = raw_field[0].cpu().permute(1, 2, 0).numpy()         # (H,W,2)
    pred_close = torch.sigmoid(raw_logits[0, 0]).cpu().numpy()       # (H,W)
    return pred_field, pred_close
