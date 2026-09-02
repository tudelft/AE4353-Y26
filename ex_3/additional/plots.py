import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_reconstructions(model, data_loader, device, n=10):
    model.eval()
    x, _ = next(iter(data_loader))
    x = x.to(device).view(-1, 28*28)

    with torch.no_grad():
        x_hat, _, _ = model(x)

    x = x.view(-1, 1, 28, 28).cpu()
    x_hat = x_hat.view(-1, 1, 28, 28).cpu()

    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    for i in range(n):
        axes[0, i].imshow(x[i][0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i][0], cmap="gray")
        axes[1, i].axis("off")
    plt.show()