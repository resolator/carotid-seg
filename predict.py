#!/usr/bin/env python3
"""Generate predictions using trained segmentation model."""
import torch
import argparse

import numpy as np
import segmentation_models_pytorch as smp

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from carotid_dataset import CarotidDataset


def parse_args():
    """Arguments parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ds-path', type=Path, required=True,
                   help='Path to the dataset for evaluation.')
    p.add_argument('--model-path', type=Path, required=True,
                   help='Path to model for evaluation.')
    p.add_argument('--save-to', type=Path, required=True,
                   help='Save predicted masks to this dir.')
    p.add_argument('--bs', type=int, default=8,
                   help='Batch size.')
    p.add_argument('--workers', type=int, default=4,
                   help='Number of data loader workers.')
    p.add_argument('--device', default='cuda',
                   choices=['cpu', 'cuda'],
                   help='Evaluation device.')

    return p.parse_args()


def main(args):
    """Application entry point."""
    if args.save_to is not None:
        args.save_to.mkdir(exist_ok=True, parents=True)

    # load dataset
    ds = CarotidDataset(args.ds_path, return_names=True)
    dl = DataLoader(ds, args.bs, num_workers=args.workers)

    # load model
    ckpt = torch.load(args.model_path, map_location='cpu')
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )
    model.load_state_dict(ckpt['model'])

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # prediction loop
    for x, y_true, names in tqdm(dl, 'Predicting'):
        # make prediction
        x, y_true = x.to(device), y_true.to(device)
        y_pred = torch.sigmoid(model(x))

        # save predictions
        y_pred_img = y_pred.detach().cpu()
        for img, name in zip(y_pred_img, names):
            np_img = (img[0] * 255).numpy()
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)

            pil_img = Image.fromarray(np_img)
            pil_img.save(args.save_to / (Path(name).stem + '.png'))

    print(f'Predictions saved to {args.save_to}')


if __name__ == '__main__':
    main(parse_args())
