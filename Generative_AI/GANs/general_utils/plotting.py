import matplotlib.pyplot as plt
import torchvision.utils as vutils

def visualize_data(dataloader, save_dir):
    # 1. Grab one batch
    batch = next(iter(dataloader))      # batch is a tensor of shape [B, C, H, W], here [64, 1, 256, 256]
    print("Batch shape:", batch.shape)
    print("Dtype:", batch.dtype, "Min/Max:", batch.min().item(), batch.max().item())

    # 2. Visualize a grid of the first 16 images
    grid = vutils.make_grid(batch[:16], nrow=4, normalize=True, value_range=(-1,1))
    plt.figure(figsize=(6,6))
    plt.axis('off')
    plt.imshow(grid.permute(1,2,0), cmap='gray')
    plt.savefig(save_dir + f"Train_images_first_16", dpi=300,
    bbox_inches='tight',
    pad_inches=0)
