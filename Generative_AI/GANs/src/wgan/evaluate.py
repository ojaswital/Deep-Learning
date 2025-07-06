import torch
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from scipy.stats import ks_2samp, wasserstein_distance


def to_uint8(x):
    # x in [-1,1] float
    return ((x + 1.0) / 2.0 * 255).clamp(0, 255).to(torch.uint8)


def to_rgb299(x_uint8):
    # x_uint8: [B,1,H,W], uint8
    x3 = x_uint8.repeat(1, 3, 1, 1)  # [B,3,H,W]
    return F.interpolate(
        x3.float(), size=(299, 299), mode='bilinear', align_corners=False
    ).to(torch.uint8)


def evaluate_wgan_stats(dataloader_real, generator, device, z_dim):
    # 1. Generate fake images matching real dataset size
    num_real = len(dataloader_real.dataset)
    batch_size = dataloader_real.batch_size
    num_batches = (num_real + batch_size - 1) // batch_size

    fake_imgs = []
    generator.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake = generator(z).cpu()
            fake_imgs.append(fake)
    fake_imgs = torch.cat(fake_imgs, dim=0)[:num_real]  # trim to match

    # 2. Create DataLoader for fakes
    fake_loader = DataLoader(
        TensorDataset(fake_imgs), batch_size=batch_size, shuffle=False
    )

    # 3. Compute FID and KID
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(normalize=False).to(device)
    for real_batch in dataloader_real:
        real_u8 = to_uint8(real_batch).to(device)
        real_inp = to_rgb299(real_u8)
        fid.update(real_inp, real=True)
        kid.update(real_inp, real=True)

        # fid.update(real_batch.to(device), real=True)
    for fake_batch, in fake_loader:
        fake_u8 = to_uint8(fake_batch).to(device)
        fake_inp = to_rgb299(fake_u8)
        fid.update(fake_inp, real=False)
        kid.update(fake_inp, real=False)

        # fid.update(fake_batch.to(device), real=False)
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_score = kid_mean.item()

    # 4. Pixel‐level KS test & Earth Mover's Distance
    # Flatten pixel values from datasets
    real_pixels = np.concatenate([batch.numpy().flatten() for batch in dataloader_real.dataset])
    fake_pixels = fake_imgs.numpy().flatten()
    ks_stat, ks_p = ks_2samp(real_pixels, fake_pixels)
    emd = wasserstein_distance(real_pixels, fake_pixels)

    # 5. Print summary
    print(f"=== GAN Statistical Evaluation ===")
    print(f"FID Score:            {fid_score:.4f}")
    print(f"KID Score:            {kid_score:.4f} (±{kid_std.item():.4f})")
    print(f"KS Statistic:         {ks_stat:.4f}, p-value: {ks_p:.4e}")
    print(f"Wasserstein Distance: {emd:.4f}")

    # Overlayed Histograms (Pixel Distributions)
    plt.figure(figsize=(6, 4))
    plt.hist(real_pixels, bins=50, alpha=0.5, label="Real", density=True)
    plt.hist(fake_pixels, bins=50, alpha=0.5, label="Fake", density=True)
    plt.legend()
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Density")
    plt.title("Pixel Distribution: Real vs. Fake")
    plt.show()

    # Q–Q Plot for Distributions - A quantile–quantile plot highlighting where your fake distribution diverges
    plt.figure(figsize=(5, 5))
    stats.probplot(fake_pixels, dist=stats.rv_histogram((np.histogram(real_pixels, bins=100, density=True))), plot=plt)
    plt.title("Q–Q Plot of Fake vs. Real Pixels")
    plt.show()

    # Bar plot
    metrics = {
        "FID": fid_score,
        "KID": kid_score,
        "KS-stat": ks_stat,
        "EMD": emd
    }
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(6, 4))
    plt.bar(names, values, color=['C0', 'C1', 'C2', 'C3'])
    plt.ylabel("Metric Value")
    plt.title("GAN Statistical Evaluation")
    plt.show()