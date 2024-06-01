#!/usr/bin/env python3
"""Train segmentation model."""
import torch
import wandb
import configargparse

import numpy as np
import segmentation_models_pytorch as smp

from tqdm import tqdm
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader
from carotid_dataset import CarotidDataset


def parse_args():
    """Arguments parser."""
    p = configargparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', is_config_file=True,
                   help='Path to config file to take arguments.')

    # data
    p.add_argument('--train-ds-path', type=Path, required=True,
                   help='Path to train dataset.')
    p.add_argument('--valid-ds-path', type=Path, required=True,
                   help='Path to validation dataset.')
    p.add_argument('--save-to', type=Path, required=True,
                   help='Save results to this directory. The name will be '
                        'used as an experiment name.')
    p.add_argument('--in-memory', action='store_true',
                   help='Load datasets into RAM.')
    p.add_argument('--workers', type=int, default=4,
                   help='Number of data loader workers.')

    # train args
    p.add_argument('--epochs', type=int, default=0,
                   help='Number of training epochs (<=0 - infinite).')
    p.add_argument('--early-stopping', type=int, default=0,
                   help='Stop training if the validation loss wasn\'t updated '
                        'this number of epochs (<=0 - disabled).')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Learning rate.')
    p.add_argument('--bs', type=int, default=8,
                   help='Batch size.')
    p.add_argument('--device', default='cuda',
                   choices=['cpu', 'cuda'],
                   help='Training device.')

    return p.parse_args()


def train_epoch(model, dls, loss_fn, optim, device):
    optim.zero_grad()
    metrics = {}
    for stage in ['train', 'valid']:
        losses = []
        is_train = stage == 'train'
        model.train() if is_train else model.eval()

        for x, y_true in tqdm(dls[stage], f'{stage}'):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y_true)
            losses.append(loss.item())

            if is_train:
                loss.backward()
                optim.step()
                optim.zero_grad()

        metrics[stage + '_loss'] = np.mean(losses)

    return metrics


def main(args):
    """Application entry point."""
    # create directories
    log_dir = args.save_to / 'logs'
    log_dir.mkdir(exist_ok=True, parents=True)
    models_dir = args.save_to / 'models'
    models_dir.mkdir(exist_ok=True, parents=True)

    # init metrics logging
    wandb.init(
        dir=log_dir,
        name=log_dir.name
    )

    # define data
    dss = {
        'train': CarotidDataset(args.train_ds_path, True, args.in_memory),
        'valid': CarotidDataset(args.valid_ds_path, False, args.in_memory),
    }
    dls = {
        'train': DataLoader(
            dss['train'],
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True
        ),
        'valid': DataLoader(
            dss['valid'],
            batch_size=args.bs,
            num_workers=args.workers
        )
    }

    # define model
    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )
    device = torch.device(args.device)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = smp.losses.dice.DiceLoss('binary')

    # train loop
    ep, update_ep = 1, 1
    epochs = args.epochs if args.epochs > 0 else np.inf
    best_valid_loss = np.inf
    while ep <= epochs:
        print(f'\nEpoch {ep}')
        metrics = train_epoch(model, dls, loss_fn, optim, device)

        # log metrics
        wandb.log(metrics, step=ep, commit=True)
        pprint(metrics)

        # update best metrics and save if better
        if metrics['valid_loss'] < best_valid_loss:
            best_valid_loss = metrics['valid_loss']
            update_ep = ep
            print(f'Updated best valid_loss to {best_valid_loss}')

            torch.save({
                'model': model.state_dict(),
                'optim': optim.state_dict()
            }, models_dir / 'valid_loss.pth')

        # check early stopping
        if args.early_stopping > 0 and ep - update_ep > args.early_stopping:
            print('Early stopping achieved.')
            break

        ep += 1
    else:
        print('Number of epochs achieved.')


if __name__ == '__main__':
    main(parse_args())
