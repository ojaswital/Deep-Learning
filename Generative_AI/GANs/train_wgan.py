import torch
import numpy as np
import yaml
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_loader import RSNATestDataset
from models import Generator256, Discriminator, gradient_penalty
from plotting import visualize_data

def train_wgan_gp(
    G,
    D,
    dataloader,
    gradient_penalty_fn,
    device,
    z_dim: int = 100,
    num_epochs: int = 50,
    lr_G: float = 3e-4,
    lr_D: float = 1e-4,
    betas: tuple = (0.5, 0.999),
    instance_noise_std: float = 0.05,
    G_steps_per_D: int = 2,
    checkpoints_folder: str = "",
    save_dir: str = ""
):
    """
    Train a WGAN-GP with optional instance noise and multiple generator steps per discriminator step.
    """
    # Ensure the checkpoint directory exists
    ckpt_dir = os.path.join(save_dir, checkpoints_folder)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Move models to device
    G.to(device)
    D.to(device)

    # Create optimizers
    optG = torch.optim.Adam(G.parameters(), lr=lr_G, betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=lr_D, betas=betas)

    lossD_vals, lossG_vals = [], []

    # Main training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        epoch_d, epoch_g = 0.0, 0.0

        for real in dataloader:
            real = real.to(device)
            B = real.size(0)

            # Add instance noise and one sided smoothening
            real_noisy = real + instance_noise_std * torch.randn_like(real)

            # ---- Discriminator update ----
            D.zero_grad()
            d_real = D(real_noisy).mean()
            z = torch.randn(B, z_dim, 1, 1, device=device)
            fake = G(z)
            fake_noisy = fake.detach() + instance_noise_std * torch.randn_like(fake)
            d_fake = D(fake_noisy).mean()
            gp = gradient_penalty_fn(D, real, fake.detach())
            lossD = d_fake - d_real + gp
            lossD.backward()
            optD.step()
            epoch_d += lossD.item()

            # ---- Generator updates ----
            # We do G_steps_per_D small updates of G for every one update of D
            for _ in range(G_steps_per_D):
                G.zero_grad()
                z = torch.randn(B, z_dim, 1, 1, device=device)
                fake = G(z)
                g_out = D(fake).mean()
                lossG = -g_out
                lossG.backward()
                optG.step()
            # accumulate once per batch
            epoch_g += lossG.item()

        # Record and save
        avgD = epoch_d / len(dataloader)
        avgG = epoch_g / len(dataloader)
        lossD_vals.append(avgD)
        lossG_vals.append(avgG)
        np.save(f"{save_dir}/{checkpoints_folder}/lossD.npy", np.array(lossD_vals))
        np.save(f"{save_dir}/{checkpoints_folder}/lossG.npy", np.array(lossG_vals))
        print(f"Epoch [{epoch}/{num_epochs}]  LossD: {avgD:.4f}  LossG: {avgG:.4f}")
        torch.save(G.state_dict(), f"{save_dir}/{checkpoints_folder}/G_epoch{epoch}.pth")

        # Plot loss
        epochs = list(range(1, len(lossD_vals) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, lossD_vals, label='Discriminator Loss')
        plt.plot(epochs, lossG_vals, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir + f"/Epoch_Loss_Generator_Discriminator.png", dpi=300,
    bbox_inches='tight',
    pad_inches=0)

    return lossD_vals, lossG_vals

def main(config_path: str):
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

    visualize_data(dataloader, os.path.join(cfg['save_dir'], cfg['checkpoints_folder']))

    # Model & training
    mdl_cfg = cfg['model']
    lossD_vals, lossG_vals = train_wgan_gp(
        G=Generator256(),
        D=Discriminator(),
        dataloader=dataloader,
        gradient_penalty_fn=gradient_penalty,
        device=device,
        z_dim=mdl_cfg['z_dim'],
        num_epochs=mdl_cfg['num_epochs'],
        lr_G=mdl_cfg['lr_G'],
        lr_D=mdl_cfg['lr_D'],
        betas=tuple(mdl_cfg['betas']),
        instance_noise_std=mdl_cfg['instance_noise_std'],
        G_steps_per_D=mdl_cfg['G_steps_per_D'],
        checkpoints_folder=mdl_cfg['checkpoints_folder'],
        save_dir=cfg['save_dir']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train WGAN-GP via config file")
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to YAML configuration file'
    )
    args = parser.parse_args()
    main(args.config)