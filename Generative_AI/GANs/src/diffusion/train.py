import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from tqdm import tqdm

from model import UNet256, Diffusion

def train_diffusion(dataloader, device, cfg):
    """
    Train a DDPM‐style diffusion model using settings from a YAML config.

    Parameters
    ----------

    Returns
    -------
    loss_vals : List[float]
        Average MSE loss per epoch.
    """

    # Prepare output directories
    save_root     = cfg['save']['save_dir']
    ckpt_folder   = os.path.join(save_root, cfg['save']['checkpoints_folder'])
    sample_folder = os.path.join(save_root, cfg['training']['sample_dir'])
    os.makedirs(ckpt_folder,   exist_ok=True)
    os.makedirs(sample_folder, exist_ok=True)

    # Instantiate diffusion helper & U‐Net
    diff_cfg = cfg['diffusion']
    diffusion = Diffusion(T=diff_cfg['T'], device=device)
    model     = UNet256(
        time_dim=cfg['model']['time_dim'],
        base_ch=cfg['model']['base_ch']
    ).to(device)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'])

    loss_vals = []

    # Training loop
    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{cfg['training']['epochs']}",
            leave=False
        )

        for real in pbar:
            real = real.to(device)
            B    = real.size(0)

            # sample random timesteps for this batch
            t = torch.randint(
                0, diffusion.T, (B,), device=device, dtype=torch.long
            )

            # compute noise‐prediction MSE loss
            loss = diffusion.p_losses(model, real, t)

            # backpropagate
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

        avg_loss = epoch_loss / len(dataloader)
        loss_vals.append(avg_loss)
        # save loss history
        np.save(os.path.join(ckpt_folder, "loss.npy"), np.array(loss_vals))
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        # save model checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_folder, f"model_epoch{epoch:03d}.pth")
        )

        # Plot loss
        epochs = list(range(1, len(loss_vals) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss_vals, label='Average Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Diffusion Model Training Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(ckpt_folder, f"Epoch_Loss_Generator_Discriminator.png"), dpi=300,
                    bbox_inches='tight',
                    pad_inches=0)

        # every sample_interval epochs, generate & save a grid
        if epoch % cfg['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                n_samples = min(64, len(dataloader.dataset))
                samples   = diffusion.sample(model, n=n_samples)
            grid = vutils.make_grid(
                samples, nrow=8, normalize=True, value_range=(-1, 1)
            )
            vutils.save_image(
                grid,
                os.path.join(sample_folder, f"epoch{epoch:03d}.png")
            )

    return loss_vals
