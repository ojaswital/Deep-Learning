import torch
import yaml
import os
import argparse
from torch.utils.data import DataLoader

from general_utils.data_loader import RSNATestDataset
from src.wgan.train import train_wgan_gp
from src.diffusion.train import train_diffusion
from general_utils.plotting import visualize_data


def main(config_path: str, model_name: str):
    # 1) Load YAML config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2) Device
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    # 3) Dataset & DataLoader
    ds_cfg = cfg['dataset']
    dataset = RSNATestDataset(ds_cfg['root'])
    dataloader = DataLoader(
        dataset,
        batch_size=ds_cfg['batch_size'],
        shuffle=ds_cfg['shuffle'],
        num_workers=ds_cfg['num_workers'],
        pin_memory=ds_cfg['pin_memory']
    )

    visualize_data(dataloader, os.path.join(cfg['save_dir'], cfg['model']['checkpoints_folder']))

    # Model & training
    if model_name == 'wgan':
        lossD_vals, lossG_vals = train_wgan_gp(
            dataloader=dataloader,
            device=device,
            cfg=cfg
        )
    elif model_name == 'diffusion':
        loss_vals = train_diffusion(
            dataloader=dataloader,
            device =device,
            cfg=cfg
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train WGAN-GP via config file")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config_gan.yaml',
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='wgan',
        help='Name of model to use - wgan or diffusion'
    )
    args = parser.parse_args()
    main(args.config, args.model)