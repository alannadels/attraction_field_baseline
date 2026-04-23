# Attraction Field Baseline — Implementation Notes

This document describes how we implemented the attraction field curve detection
method from:

> **Robust Curve Detection in Volumetric Medical Imaging via Attraction Field**
> Yaushev et al., arXiv:2408.01159v2 (2024)
> Code/data: https://github.com/neuro-ml/curve-detection

The published GitHub repository contains only **annotation data** (aortic
centerline knots for 3D CT volumes) — no model code was released.  We therefore
implemented the full method from scratch, following every equation in the paper,
then adapted it from 3D volumetric images to our 2D synthetic snake dataset.

---

## 1. Method Overview

The method predicts two pixel-wise maps from a single image:

1. **Attraction field** `F_p` — for each pixel `p`, the displacement vector pointing
   to the nearest point on the target curve.
2. **Closeness map** `C_p` — a binary label indicating whether pixel `p` lies within
   a distance `Rc` of the curve.

At inference, the two maps are combined to produce a point cloud on the curve,
which is thinned with NMS and then ordered using Isomap to yield a parameterised
curve `(x, y, dist)`.

---

## 2. Ground Truth Construction

### 2.1 Attraction Field

For every pixel `p` in an image with ground-truth curve `Γ`, let `r_p` be the
nearest point on `Γ` to `p`.  The ground-truth attraction vector is:

```
F_p = r_p − p
```

The norm `‖F_p‖₂` is the Euclidean distance from pixel `p` to the curve.

**Implementation** (`af_dataset.py`, `_compute_gt`):

```python
tree = KDTree(curve_pts)                    # curve_pts: (N, 2) ordered pixel coords
pixel_xy = np.stack([xs.ravel(), ys.ravel()], axis=1)   # all (x, y) pixel positions
dists, idxs = tree.query(pixel_xy)

nearest_xy   = curve_pts[idxs]             # r_p for every pixel
field_flat   = nearest_xy - pixel_xy       # F_p = r_p − p  for every pixel
```

### 2.2 Closeness Map

```
C_p = 𝟙[‖F_p‖₂ ≤ Rc]
```

where `Rc` is a hyperparameter controlling the radius of interest around the
curve.  Pixels beyond `Rc` are treated as background.

```python
closeness = (dists <= self.Rc).astype(np.float32)
```

---

## 3. Architecture

Paper 1 uses a **3D VNet** backbone with two output heads, each consisting of
six residual blocks (kernel 3×3×3, padding 1).

Since our data is 2D, we replace the 3D VNet with a **2D UNet** encoder-decoder.
The two-head structure and the six-residual-block design per head are preserved
exactly.

```
Input (1×H×W)
    │
    ├─ Encoder: 4 stages of ConvBlock + MaxPool2d
    ├─ Bottleneck: ConvBlock
    └─ Decoder: 4 stages of ConvTranspose2d + skip-cat + ConvBlock
         │
         ├─ Field head:     6 ResBlocks → Conv2d(→ 2 ch)   [dx, dy]
         └─ Closeness head: 6 ResBlocks → Conv2d(→ 1 ch)   [logit]
```

**Implementation** (`af_model.py`):

```python
class ResBlock(nn.Module):                  # mirrors VNet residual block
    def __init__(self, channels):
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
    def forward(self, x):
        return nn.ReLU()(x + self.block(x))

class OutputHead(nn.Module):               # 6 ResBlocks + 1×1 projection
    def __init__(self, in_ch, out_ch, n_res=6):
        layers = [ResBlock(in_ch) for _ in range(n_res)]
        layers.append(nn.Conv2d(in_ch, out_ch, 1))
        self.net = nn.Sequential(*layers)
```

The field head outputs raw (unbounded) `(dx, dy)` vectors.
The closeness head outputs a raw logit; sigmoid is applied at inference.

---

## 4. Loss Function

The total loss is the unweighted sum of three terms (Paper 1, Eq. 4):

```
L = L_field + L_cls + L_norm
```

### 4.1 Attraction Field Loss (Eq. 1)

```
L_field(F, F̂) = (1/N) Σ_p  ‖F̂_p − F_p‖₁
```

Only computed for pixels within `Rf` of the curve (to avoid the
multiple-projections ambiguity — pixels far from the curve may project to
several segments, making the field direction undefined).

```python
def loss_field(pred_field, gt_field, dist_map, Rf):
    mask = (dist_map <= Rf).float()             # restrict to ||F_p||₂ ≤ Rf
    mask2 = mask.expand_as(pred_field)
    err = (pred_field - gt_field).abs() * mask2
    return err.sum() / mask2.sum().clamp(min=1)
```

### 4.2 Closeness Loss (Eq. 2)

```
L_cls(C, Ĉ) = (1/N) Σ_p  BCE(Ĉ_p, C_p)
```

where `C_p = 𝟙[‖F_p‖₂ ≤ Rc]`.

```python
def loss_cls(pred_logits, gt_closeness):
    return F.binary_cross_entropy_with_logits(pred_logits, gt_closeness)
```

### 4.3 Norm Regularisation Loss (Eq. 3)

The norm regulariser addresses the multiple-projections problem: if a pixel
has several equidistant projections onto the curve, the network's field head
is likely to predict a near-zero vector (averaging the contradictory directions).
Penalising the difference in **norms** only — not directions — gives a
consistent training signal regardless of directional ambiguity:

```
L_norm(F, F̂) = (1/N) Σ_p  | ‖F̂_p‖₂ − ‖F_p‖₂ |
```

Again restricted to pixels within `Rf`.

```python
def loss_norm(pred_field, gt_field, dist_map, Rf):
    mask = (dist_map <= Rf).float()
    pred_norm = pred_field.norm(dim=1, keepdim=True)
    gt_norm   = gt_field.norm(dim=1, keepdim=True)
    err = (pred_norm - gt_norm).abs() * mask
    return err.sum() / mask.sum().clamp(min=1)
```

---

## 5. Inference Pipeline

At test time the model runs through five steps (Paper 1, Section 2.2).

### Step 1 — Forward Pass

```
F̂, Ĉ_logit = model(image)
Ĉ = sigmoid(Ĉ_logit)
```

### Step 2 — Point Cloud Extraction (Eq. 5)

Only pixels satisfying both conditions contribute to the point cloud:

```
Point cloud = { F̂_p + p  |  Ĉ_p ≥ t,  ‖F̂_p‖₂ ≤ Rf }
```

The `Ĉ_p ≥ t` condition (closeness head) suppresses false positives far
from the curve.  The `‖F̂_p‖₂ ≤ Rf` condition (field norm) improves recall
by excluding pixels whose field predictions are unreliable.

```python
norms = np.linalg.norm(pred_field, axis=2)
mask  = (pred_closeness >= t) & (norms <= Rf)
projected = pixel_xy[mask] + pred_field[mask]   # shift pixel → curve
```

### Step 3 — Non-Maximum Suppression

The paper uses NMS with the Euclidean distance as neighbourhood function and
`confidence = −‖F̂_p‖₂` (smaller norm = more confident prediction, since the
pixel is geometrically closer to the curve).

```python
# Sort by ascending norm (= descending confidence)
order = np.argsort(norms)
# Greedy: keep point, suppress all neighbours within radius
for i in range(len(pts)):
    dists_to_rest = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
    kept_mask[i+1:][dists_to_rest < radius] = False
```

### Step 4 — Isomap Ordering

The thinned point cloud is an unordered set of 2D points near the curve.
To recover a 1-D parameterisation (the `dist` value used by our evaluation
metric), we apply **Isomap** dimensionality reduction to 1 component:

```
dist = Isomap(n_components=1).fit_transform(cloud)   # geodesic 1-D embedding
dist = (dist − min) / (max − min)                    # normalise to [0, 1]
```

Isomap preserves geodesic distances along the manifold, making it robust to
the curve's local geometry and self-proximity (e.g. a snake that doubles back
on itself).

```python
iso = Isomap(n_components=1, n_neighbors=k)
emb = iso.fit_transform(points).ravel()
dist = (emb - emb.min()) / (emb.max() - emb.min())
```

---

## 6. Evaluation Metric

We evaluate using the **dense-curve Chamfer distance** from Our Paper
(§5.1 and §6.5) rather than Paper 1's HD/ASSD/SD metrics, so that results
are directly comparable to our method's variants (v2/v8/v9/v10).

### 6.1 Densification Protocol

Both predicted and target curves are densified before comparison:

1. Sort control points by `dist`.
2. Interpolate `(x, y)` with a **Catmull-Rom spline**.
3. Linearly interpolate `dist` along the arclength of the dense curve.
4. This yields a dense 3-D cloud in `[x, y, d]` space.

```python
dense_xy  = catmull_rom(ctrl_xy, n_per_seg=20)          # (M, 2)
dense_arc = arclength_normalised(dense_xy)              # arc param in [0,1]
dense_d   = np.interp(dense_arc, sparse_arc, ctrl_d)   # d interpolated
dense_xyd = np.concatenate([dense_xy, dense_d[:,None]], axis=1)
```

### 6.2 Symmetric Chamfer Distance (L1, p=1)

```
C(Â, B̂) = 1/2 [ (1/|Â|) Σ_{a∈Â} min_{b∈B̂} ‖a−b‖₁
               + (1/|B̂|) Σ_{b∈B̂} min_{a∈Â} ‖b−a‖₁ ]
```

All `(x, y)` coordinates are normalised to `[0, 1]` by dividing by
`image_size = 400`, so that spatial and dist dimensions are on comparable
scales.

```python
diff    = np.abs(A[:,None,:] - B[None,:,:]).sum(axis=2)  # (M, N) pairwise L1
a_to_b  = diff.min(axis=1).mean()
b_to_a  = diff.min(axis=0).mean()
chamfer = 0.5 * (a_to_b + b_to_a)
```

---

## 7. Hyperparameter Adaptation (3D → 2D)

Paper 1 operates on 3D CT volumes resampled to 2×2×2 mm³, so its hyperparameters
are in mm.  We convert proportionally to pixel units for our 400×400 images.

| Hyperparameter | Paper 1 (3D) | This implementation (2D) | Role |
|---|---|---|---|
| `Rc` | 10 mm (5 voxels) | **15 px** | Closeness radius |
| `Rf` | 5 mm (2.5 voxels) | **8 px** | Field training radius |
| `t` | 0.5 | **0.5** | Closeness threshold at inference |
| NMS radius | not specified | **3 px** | Point cloud thinning |
| Isomap `k` | not specified | **15** | Neighbourhood for geodesic graph |
| Batch size | 2 | **2** | Kept identical |
| LR schedule | 1e-4 → 5e-5 → 5e-6 | **1e-4 → 1e-5 → 1e-6** | Proportional decay |

---

## 8. What Changed vs. What Stayed the Same

| Aspect | Paper 1 | This implementation | Reason for change |
|---|---|---|---|
| **Backbone** | 3D VNet | 2D UNet | Data is 2D |
| **Convolutions** | 3D (k=3, p=1) | 2D (k=3, p=1) | Data is 2D |
| **Head design** | 6 res-blocks per head | 6 res-blocks per head | Identical |
| **Loss L_field** | Eq. 1 | Identical | — |
| **Loss L_cls** | Eq. 2 | Identical | — |
| **Loss L_norm** | Eq. 3 | Identical | — |
| **Total loss** | Eq. 4 | Identical | — |
| **Point cloud** | Eq. 5 | Identical | — |
| **NMS** | Section 2.2 | Identical | — |
| **Isomap ordering** | Section 2.2 | Identical | — |
| **Evaluation metric** | HD / ASSD / SD | Dense Chamfer [x,y,d] | Match our paper |

---

## 9. File Reference

| File | Contents |
|---|---|
| `af_dataset.py` | `SnakeCurveDataset` — KDTree ground-truth field/closeness computation |
| `af_model.py` | `AttractionFieldNet` — 2D UNet + two 6-ResBlock heads |
| `af_loss.py` | `loss_field`, `loss_cls`, `loss_norm`, `total_loss` (Eq. 1–4) |
| `af_inference.py` | `extract_point_cloud`, `nms`, `order_with_isomap`, `predict_curve` |
| `af_evaluate.py` | `catmull_rom`, `chamfer_l1`, `evaluate_one`, `evaluate_dataset` |
| `af_train.py` | Training loop, Adam optimiser, LR schedule, mid-training validation |
| `main.py` | CLI entry point: `train`, `eval`, `test` modes |

## 10. Usage

```bash
# Train (uses GPU if available):
python main.py --mode train

# Evaluate best checkpoint on full validation split:
python main.py --mode eval --checkpoint outputs/checkpoints/best_model.pth

# Quick smoke test (2 epochs, CPU, small model):
python main.py --mode test
```
