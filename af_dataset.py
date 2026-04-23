"""
Dataset for Paper 1 (Attraction Field) method adapted to 2D snake curves.

Paper 1: "Robust Curve Detection in Volumetric Medical Imaging via Attraction Field"
Adaptation: 3D volumetric → 2D grayscale images; voxels → pixels.

Ground truth per image:
  - Attraction field F_p = (nearest_curve_point - pixel_p), shape (H, W, 2)
  - Closeness map C_p = 1 if ||F_p||_2 <= Rc else 0, shape (H, W)
  - Distance map ||F_p||_2, shape (H, W)   [used for loss masking]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.spatial import KDTree


class SnakeCurveDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        indices: list,
        image_size: int = 400,
        Rc: float = 15.0,
        Rf: float = 8.0,
    ):
        """
        Args:
            image_dir: path to images/ directory
            label_dir: path to labels/ directory
            indices:   list of integer curve indices (1-based, e.g. [1, 2, ..., 4870])
            image_size: H=W of loaded image (400 = native; labels stay in pixel coords)
            Rc:  closeness radius (pixels) — C_p=1 iff dist <= Rc
            Rf:  field training radius (pixels) — Lfield only within Rf
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.indices = indices
        self.image_size = image_size
        self.Rc = Rc
        self.Rf = Rf

        # Pre-build pixel grid (shared across items)
        H = W = image_size
        ys, xs = np.mgrid[0:H, 0:W]           # (H, W) each
        # pixel_coords[r, c] = (x=c, y=r)
        self._px = xs.ravel().astype(np.float32)   # x = col
        self._py = ys.ravel().astype(np.float32)   # y = row

    # ------------------------------------------------------------------
    def _compute_gt(self, curve_pts: np.ndarray):
        """
        curve_pts: (N, 2) float64 array of (x, y) pixel coords along curve.

        Returns
        -------
        field      : (H, W, 2)  float32  attraction vectors (dx, dy)
        closeness  : (H, W)     float32  binary {0, 1}
        dist_map   : (H, W)     float32  distance to nearest curve point
        """
        H = W = self.image_size
        px, py = self._px, self._py                # (H*W,)

        tree = KDTree(curve_pts)                   # curve_pts in (x, y) space
        pixel_xy = np.stack([px, py], axis=1)      # (H*W, 2)
        dists, idxs = tree.query(pixel_xy)

        nearest_xy = curve_pts[idxs]               # (H*W, 2) nearest (x, y)
        field_flat = (nearest_xy - pixel_xy).astype(np.float32)   # (H*W, 2)

        dists = dists.astype(np.float32)

        field    = field_flat.reshape(H, W, 2)
        closeness = (dists <= self.Rc).astype(np.float32).reshape(H, W)
        dist_map  = dists.reshape(H, W)

        return field, closeness, dist_map

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        name = f'curve_{i:04d}'

        # ---- image -------------------------------------------------------
        img_path = os.path.join(self.image_dir, f'{name}.jpeg')
        img = Image.open(img_path).convert('L')        # grayscale
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)   # (1, H, W)

        # ---- label -------------------------------------------------------
        lbl_path = os.path.join(self.label_dir, f'{name}.npy')
        curve_pts = np.load(lbl_path)                  # (N, 2) float64

        # ---- ground truth fields -----------------------------------------
        field, closeness, dist_map = self._compute_gt(curve_pts)

        field_tensor    = torch.from_numpy(field).permute(2, 0, 1)    # (2, H, W)
        close_tensor    = torch.from_numpy(closeness).unsqueeze(0)    # (1, H, W)
        dist_tensor     = torch.from_numpy(dist_map).unsqueeze(0)     # (1, H, W)

        return img_tensor, field_tensor, close_tensor, dist_tensor, curve_pts


# ------------------------------------------------------------------
def collate_fn(batch):
    """Custom collate to handle variable-length curve_pts."""
    imgs       = torch.stack([b[0] for b in batch])
    fields     = torch.stack([b[1] for b in batch])
    closeness  = torch.stack([b[2] for b in batch])
    dist_maps  = torch.stack([b[3] for b in batch])
    curve_pts_list = [b[4] for b in batch]           # list of (N_i, 2) arrays
    return imgs, fields, closeness, dist_maps, curve_pts_list


# ------------------------------------------------------------------
def build_splits(n_total=4870, val_frac=0.2, seed=42):
    """Return (train_indices, val_indices) as lists of 1-based integers."""
    rng = np.random.default_rng(seed)
    indices = np.arange(1, n_total + 1)
    rng.shuffle(indices)
    n_val = int(n_total * val_frac)
    val_idx   = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return train_idx, val_idx
