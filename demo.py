#!/usr/bin/env python3
"""
This script generates a segmentation mask on an input ultrasound carotid
image, draws and saves it.
"""
import cv2
import torch
import argparse

import numpy as np
import segmentation_models_pytorch as smp
import torchvision.transforms.v2 as transforms

from PIL import Image
from pathlib import Path


def parse_args():
    """Arguments parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-path', type=Path, required=True,
                   help='Path to an image for demonstration.')
    p.add_argument('--model-path', type=Path, required=True,
                   help='Path to a model for evaluation.')
    p.add_argument('--save-to', type=Path, required=True,
                   help='Path to a resulted image.')
    p.add_argument('--seg-mask', action='store_true',
                   help='Visualize a segmentation mask instead of bbox.')
    p.add_argument('--enlarge', type=float, default=0.25,
                   help='Enlarge the bbox by this factor.')

    return p.parse_args()


def main(args):
    """Application entry point."""
    assert args.enlarge >= 0, 'enlarge can\'t be lower 0'
    assert args.image_path.exists(), 'wrong image_path'
    assert args.model_path.exists(), 'wrong model_path'
    assert args.model_path.suffix == '.pth', \
        'model_path should be a path to saved .pth model'

    assert args.save_to.parent.exists(), \
        'save_to parent folder doesn\'t exists'

    allowed_formats = {'.jpg', '.png'}
    assert args.save_to.suffix.lower() in allowed_formats, \
        f'save_to image should one of the following formats: {allowed_formats}'

    # load model
    ckpt = torch.load(args.model_path, map_location='cpu')
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )
    model.load_state_dict(ckpt['model'])
    model.eval()

    # prepare sample
    src_img = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    resize_to = 256  # TODO: read from ckpt
    img_mean = torch.tensor([0.1142])  # TODO: read from ckpt
    img_std = torch.tensor([0.1621])  # TODO: read from ckpt
    img_infer_t = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize(
            resize_to,
            max_size=resize_to + 1
        ),
        transforms.CenterCrop(resize_to),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std)
    ])

    # make prediction
    x = img_infer_t(img)
    y_pred = torch.sigmoid(model(x[None])).detach().cpu()[0, 0]

    # find the biggest blob
    th = 0.5  # TODO: read from model
    pd_mask = ((y_pred > th) * 255).numpy().astype(np.uint8)
    contours, _ = cv2.findContours(
        pd_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    max_contour = max(contours, key=cv2.contourArea)

    # prepare visualization
    pd_mask = np.zeros_like(pd_mask)
    pd_mask = cv2.drawContours(pd_mask, [max_contour], -1, (255), -1)

    # resize to an input image
    pd_mask = cv2.resize(
        pd_mask,
        (src_img.shape[1], src_img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    if args.seg_mask:
        masked = src_img.copy() * 0.75
        masked[:, :, 1] = cv2.add(pd_mask * 0.25, masked[:, :, 1])
        visualized = np.hstack([
            src_img,
            cv2.cvtColor(pd_mask, cv2.COLOR_GRAY2RGB),
            masked
        ]).astype(np.uint8)
    else:
        x, y, w, h = cv2.boundingRect(pd_mask)

        # enlarge
        w_e = int(w * args.enlarge // 2)
        h_e = int(h * args.enlarge // 2)
        x -= w_e
        y -= h_e
        w += w_e * 2
        h += h_e * 2
        cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        visualized = src_img

    pil_img = Image.fromarray(visualized)
    pil_img.save(args.save_to)
    print(f'Visualization saved to {args.save_to.absolute()}')


if __name__ == '__main__':
    main(parse_args())
