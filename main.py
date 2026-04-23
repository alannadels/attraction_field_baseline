"""
Main entry point for the Attraction Field baseline (Paper 1 adaptation).

Usage
-----
# Train (default config):
    python main.py --mode train

# Train with custom settings:
    python main.py --mode train --epochs 100 --batch_size 4 --device cuda

# Evaluate a saved checkpoint on the validation split:
    python main.py --mode eval --checkpoint outputs/checkpoints/best_model.pth

# Quick smoke-test (5 images, CPU):
    python main.py --mode test
"""

import argparse
import os

import numpy as np
import torch

# --------------------------------------------------------------------------
DATASET_ROOT = '/Users/alannadels/Desktop/datasets/synthetic_image_grayscale_dataset'
IMAGE_DIR    = os.path.join(DATASET_ROOT, 'images')
LABEL_DIR    = os.path.join(DATASET_ROOT, 'labels')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'outputs')


# --------------------------------------------------------------------------
def get_cfg(args) -> dict:
    from af_train import DEFAULT_CFG
    cfg = DEFAULT_CFG.copy()

    # Override from CLI
    if args.epochs is not None:
        cfg['n_epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.device is not None:
        cfg['device'] = args.device
    if args.lr is not None:
        cfg['lr'] = args.lr

    return cfg


# --------------------------------------------------------------------------
def run_train(args):
    from af_train import train
    cfg = get_cfg(args)

    print('=== Attraction Field Training ===')
    print(f'Device : {cfg["device"]}')
    print(f'Epochs : {cfg["n_epochs"]}  |  Batch: {cfg["batch_size"]}  |  LR: {cfg["lr"]}')
    print(f'Rc={cfg["Rc"]}px  Rf={cfg["Rf"]}px  t={cfg["t_thresh"]}')
    print()

    train(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR, cfg)


# --------------------------------------------------------------------------
def run_eval(args):
    from af_train import DEFAULT_CFG
    from af_dataset import build_splits, SnakeCurveDataset
    from af_model import AttractionFieldNet
    from af_inference import model_predict, predict_curve
    from af_evaluate import evaluate_dataset

    cfg = get_cfg(args)
    device = cfg['device']

    # Load model
    model = AttractionFieldNet(
        in_ch=1, base_ch=cfg['base_ch'], head_res=cfg['head_res'],
    ).to(device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'best_model.pth')
    print(f'Loading checkpoint: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Val split (same seed → same split as training)
    _, val_idx = build_splits(n_total=4870, val_frac=cfg['val_frac'], seed=cfg['seed'])
    print(f'Evaluating on {len(val_idx)} validation images…')

    val_ds = SnakeCurveDataset(
        IMAGE_DIR, LABEL_DIR, val_idx,
        image_size=cfg['image_size'], Rc=cfg['Rc'], Rf=cfg['Rf'],
    )

    predictions, gt_curves = [], []
    for i in range(len(val_ds)):
        img_t, _, _, _, gt_pts = val_ds[i]
        img_t = img_t.unsqueeze(0).to(device)

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

        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(val_ds)}', flush=True)

    results = evaluate_dataset(
        predictions, gt_curves,
        image_size=cfg['image_size'],
        n_per_seg=cfg['n_per_seg'],
    )

    print('\n=== Evaluation Results ===')
    print(f'Mean Chamfer Distance : {results["mean_chamfer"]:.6f}')
    print(f'Failed predictions    : {results["n_failed"]} / {len(val_idx)}')

    scores = [s for s in results['per_sample'] if s != float('inf')]
    if scores:
        print(f'Median Chamfer        : {np.median(scores):.6f}')
        print(f'Std Chamfer           : {np.std(scores):.6f}')


# --------------------------------------------------------------------------
def run_test(args):
    """Quick smoke-test: 5 images, CPU, minimal epochs."""
    from af_train import train, DEFAULT_CFG
    cfg = DEFAULT_CFG.copy()
    cfg.update({
        'n_epochs':    2,
        'val_every':   1,
        'batch_size':  2,
        'num_workers': 0,
        'device':      'cpu',
        'base_ch':     16,
        'head_res':    2,
    })
    print('=== Smoke test (2 epochs, CPU, small model) ===')
    train(IMAGE_DIR, LABEL_DIR, os.path.join(OUTPUT_DIR, 'smoke_test'), cfg)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Attraction Field baseline for DLO centerline detection'
    )
    parser.add_argument('--mode', choices=['train', 'eval', 'test'], default='train')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth file (for eval mode)')
    parser.add_argument('--epochs',     type=int,   default=None)
    parser.add_argument('--batch_size', type=int,   default=None)
    parser.add_argument('--lr',         type=float, default=None)
    parser.add_argument('--device',     type=str,   default=None,
                        help='e.g. cuda, cuda:0, cpu')
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'eval':
        run_eval(args)
    elif args.mode == 'test':
        run_test(args)


if __name__ == '__main__':
    main()
