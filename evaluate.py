#!/usr/bin/env python3
"""Evaluate segmentation predictions."""
import cv2
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path


def parse_args():
    """Arguments parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gt-masks-dir', type=Path, required=True,
                   help='Path to ground-truth masks dir.')
    p.add_argument('--pd-masks-dir', type=Path, required=True,
                   help='Path to predicted masks dir.')
    p.add_argument('--th', type=float, default=0.5,
                   help='Binarization threshold.')

    return p.parse_args()


def main(args):
    """Application entry point."""
    ious = []
    for gt_mask_p in tqdm(list(args.gt_masks_dir.glob('*.*')), 'Evaluating'):
        # data reading
        gt_mask = cv2.imread(str(gt_mask_p), cv2.IMREAD_GRAYSCALE)
        pd_mask_p = args.pd_masks_dir / gt_mask_p.name
        pd_mask = cv2.imread(str(pd_mask_p), cv2.IMREAD_GRAYSCALE)

        # find the biggest blob
        pd_mask = ((pd_mask > args.th) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            pd_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        max_contour = max(contours, key=cv2.contourArea)
        pd_mask = np.zeros_like(pd_mask)
        pd_mask = cv2.drawContours(pd_mask, [max_contour], -1, (255), -1)

        # resize prediction to gt
        pd_mask = cv2.resize(
            pd_mask,
            (gt_mask.shape[1],
             gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # calculating iou
        intersection = (gt_mask & pd_mask).sum()
        union = (gt_mask | pd_mask).sum()
        ious.append(intersection / union)

    print('Mean IoU:', np.mean(ious))


if __name__ == '__main__':
    main(parse_args())
