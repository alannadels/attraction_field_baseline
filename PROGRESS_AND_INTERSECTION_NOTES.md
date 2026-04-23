# Progress Notes — Attraction Field Baseline

---

## 1. What Has Been Achieved

The goal of this workstream is to take Paper 1 (Yaushev et al., "Robust Curve
Detection in Volumetric Medical Imaging via Attraction Field", arXiv 2408.01159)
and get it running on our 2D synthetic snake dataset under our evaluation
conditions.

**First finding:** The paper's GitHub repository (https://github.com/neuro-ml/curve-detection)
contains only annotation data for 3D CT volumes — no model code was released.
The method was therefore implemented in full from scratch, following every
equation in the paper.

**What was built:**

| Component | Description |
|---|---|
| Ground truth pipeline | For each pixel in a training image, a KDTree computes the nearest curve point, yielding a per-pixel attraction vector `F_p = r_p − p` and a binary closeness label `C_p = 𝟙[‖F_p‖₂ ≤ Rc]` |
| Model | 2D UNet encoder-decoder with two output heads (attraction field + closeness), each with 6 residual blocks — directly mirroring Paper 1's two-head VNet design, with 3D convolutions replaced by 2D |
| Loss | Full three-term loss: `L = L_field + L_cls + L_norm` (Paper 1, Eq. 1–4), implemented exactly |
| Inference | Point cloud extraction (Paper 1, Eq. 5) → greedy NMS → Isomap 1D ordering |
| Evaluation | Dense-curve symmetric Chamfer distance in `[x, y, d]` space using Catmull-Rom densification, matching our paper's evaluation protocol (§5.1, §6.5) rather than Paper 1's medical metrics (HD/ASSD/SD) |
| Training script | Adam optimizer, LR decay schedule mirroring Paper 1, checkpointing, mid-training validation on a held-out subset |
| CLI | `python main.py --mode train/eval/test` |

The pipeline has been verified end-to-end: data loading, loss computation,
forward pass, inference, and Chamfer evaluation all run correctly. The code is
ready to train on a GPU.

---

## 2. Does the Method Handle Intersections and Overlaps?

**No — not in any meaningful sense.**

Paper 1 was designed for anatomical curves in medical imaging (aortic centerlines,
vertebral columns) that are non-self-intersecting by nature. The method has no
mechanism for reasoning about curve identity across branches or occlusion
boundaries.

The paper does acknowledge one related problem — **multiple projections** — which
occurs when a pixel is equidistant to two separate segments of the curve. Their
mitigation is:

- **L_norm (Eq. 3):** Even when a pixel's predicted direction vector averages
  to near-zero due to directional ambiguity between two segments, the distance
  to the curve is still well-defined and can be supervised correctly.
- **Rf restriction:** The field loss is only computed within `Rf` pixels of the
  curve, reducing the chance that any training pixel sits ambiguously between
  two segments.

However, this is not the same as handling true self-intersection. At each stage
of the pipeline, self-intersection causes a distinct and unresolved failure:

- **Ground truth:** The KDTree assigns each pixel to whichever branch is
  marginally closer. Neighboring pixels near a crossing may be assigned to
  different branches, creating a field that flips direction abruptly — an
  inconsistency that L_norm cannot resolve.

- **Point cloud:** Pixels near the intersection project onto both branches
  indiscriminately. The resulting cloud is a mixed, interleaved set of points
  with no structural separation between the two crossing segments.

- **Isomap ordering:** This is the most severe failure point. Isomap recovers
  a 1D parameterization via geodesic distances on the local neighborhood graph.
  At a true crossing, the graph connects points from two different branches,
  and Isomap cannot distinguish traveling along one branch from crossing to the
  other. The output `dist` values near any self-intersection are expected to be
  incorrect, producing mis-ordered or folded curve predictions at crossings.

In short: the method treats the curve as a simple, non-branching, non-crossing
1D manifold at every stage. Self-intersection breaks this assumption at every
stage.

---

## 3. Possible Extensions to Handle Intersections and Overlaps

Below are three credible directions, ordered roughly by implementation
complexity.

---

### Option A — Crossing-Aware Post-Processing on the Point Cloud

**Idea:** Leave the model and training unchanged. After NMS produces the thinned
point cloud, run a post-processing step that detects crossing regions and
locally repairs the ordering before Isomap is applied.

**How it could work:**

1. Detect high-density clusters in the point cloud — regions where far more
   points than expected for a single pass of the curve are concentrated in a
   small area. These are candidate crossing zones.
2. Within each crossing zone, fit two local line segments (e.g., via RANSAC or
   a short PCA sweep) to separate the two branches.
3. Re-assign points to branches and run Isomap on each clean sub-cloud
   separately, then stitch the `dist` parameterizations.

**Pros:** No retraining. Purely algorithmic.

**Cons:** Heuristic and brittle for complex crossings. Fails if the two
branches are not locally distinguishable by direction (e.g., near-perpendicular
crossing is easy; near-parallel overlap is not).

---

### Option B — Additional "Crossing Map" Head

**Idea:** Add a third output head to the model that predicts a binary
**crossing map** — `1` at pixels that lie within a crossing or overlap region,
`0` elsewhere. This gives the inference pipeline explicit spatial knowledge
of where crossings occur.

**How it could work:**

- **Ground truth:** A pixel is labeled as a crossing if it lies within `Rc` of
  two or more distinct segments of the curve (i.e., it has at least two
  projections within `Rc`). This is trivially detectable from the KDTree during
  the existing GT computation step — any pixel whose `k`-nearest curve points
  span a geodesic distance greater than some threshold along the curve are in a
  crossing region.
- **Loss:** Standard BCE on the crossing map, added to the existing total loss.
- **Inference:** Mask out crossing regions from the initial point cloud.
  Handle each branch on either side of the crossing separately, then reconnect
  them using the predicted crossing location as a bridge point.

**Pros:** Principled; the model learns where crossings are directly from the
data. Relatively low additional architectural complexity (one extra head).

**Cons:** Requires retraining. The reconnection logic at inference is still
heuristic, and for tightly overlapping (parallel) segments, the crossing map
alone may not be sufficient to separate the branches.

---

### Option C — Branch-Indexed Attraction Field

**Idea:** The most principled change: replace the single attraction field with
**K branch-indexed fields**, one per curve segment between consecutive
intersection points. Each head predicts the attraction field for one
topological segment of the curve. This gives the model the ability to
simultaneously represent multiple overlapping segments at a single pixel.

**How it could work:**

- **Ground truth:** Use the curve's critical points (already available in our
  dataset as `crit_points/`) to segment the curve into `K` topological branches
  at each crossing. Each pixel is assigned a field vector per branch, with a
  validity flag indicating which branches are close enough to supervise.
- **Model:** `K` parallel field heads sharing the same UNet encoder, producing
  `K × 2` channel field outputs and `K` closeness outputs.
- **Inference:** Extract `K` independent point clouds, one per branch. Each
  cloud is ordered with Isomap independently. The branches are then concatenated
  into the full ordered curve by matching endpoints.

**Pros:** Handles true multi-branch ambiguity without heuristic post-processing.
Architecturally clean.

**Cons:** Requires knowing `K` (max number of simultaneous crossings) in
advance. Ground truth construction is more complex. Substantially increases
model size and training complexity. For a baseline comparison method, this
moves significantly away from Paper 1's original design.

---

## Recommendation

For the purposes of this paper, **Option A** is the most appropriate near-term
choice. It keeps the implementation faithful to Paper 1 (important for a fair
comparison baseline) while giving the method a fighting chance on self-intersecting
curves. Options B and C represent legitimate future research directions but
would effectively make this a new method rather than a baseline.

The more useful framing for the paper may simply be to **report the failure
mode honestly**: Paper 1 achieves competitive performance on smooth, non-crossing
portions of snake curves but degrades at self-intersections — which is precisely
the failure mode our method (v8/v9/v10) is designed to overcome.
