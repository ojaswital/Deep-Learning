import torch
import numpy as np
import yaml
import os
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import Generator256, Discriminator, gradient_penalty

def train_wgan_gp(dataloader, device, cfg):
    """
    Train a WGAN-GP with optional instance noise and multiple generator steps per discriminator step.
    """
    # Ensure the checkpoint directory exists
    ckpt_dir = os.path.join(cfg['save']['save_dir'], cfg['save']['checkpoints_folder'])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Move models to device
    G = Generator256()
    D = Discriminator()
    G.to(device)
    D.to(device)

    # Create optimizers
    optG = torch.optim.Adam(G.parameters(), lr=cfg['model']['lr_G'], betas=betas)
    optD = torch.optim.Adam(D.parameters(), lr=cfg['model']['lr_D'], betas=betas)

    lossD_vals, lossG_vals = [], []

    # Main training loop
    for epoch in range(1, cfg['model']['num_epochs'] + 1):
        print(f"Epoch {epoch}/{cfg['model']['num_epochs']}")
        epoch_d, epoch_g = 0.0, 0.0

        for real in dataloader:
            real = real.to(device)
            B = real.size(0)

            # Add instance noise and one sided smoothening
            real_noisy = real + cfg['model']['instance_noise_std'] * torch.randn_like(real)

            # ---- Discriminator update ----
            D.zero_grad()
            d_real = D(real_noisy).mean()
            z = torch.randn(B, cfg['model']['z_dim'], 1, 1, device=device)
            fake = G(z)
            fake_noisy = fake.detach() + cfg['model']['instance_noise_std'] * torch.randn_like(fake)
            d_fake = D(fake_noisy).mean()
            gp = gradient_penalty(D, real, fake.detach())
            lossD = d_fake - d_real + gp
            lossD.backward()
            optD.step()
            epoch_d += lossD.item()

            # ---- Generator updates ----
            # We do G_steps_per_D small updates of G for every one update of D
            for _ in range(cfg['model']['G_steps_per_D']):
                G.zero_grad()
                z = torch.randn(B, cfg['model']['z_dim'], 1, 1, device=device)
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
        np.save(f"{ckpt_dir}/lossD.npy", np.array(lossD_vals))
        np.save(f"{ckpt_dir}/lossG.npy", np.array(lossG_vals))
        print(f"Epoch [{epoch}/{cfg['model']['num_epochs']}]  LossD: {avgD:.4f}  LossG: {avgG:.4f}")
        torch.save(G.state_dict(), os.path.join(ckpt_dir,f"G_epoch{epoch}.pth"))

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
        plt.savefig(os.path.join(ckpt_dir, f"Epoch_Loss_Generator_Discriminator.png"), dpi=300,
        bbox_inches='tight',
        pad_inches=0)

    return lossD_vals, lossG_vals