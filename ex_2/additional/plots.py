import matplotlib.pyplot as plt
import torch
import random

import matplotlib.pyplot as plt
import random
import numpy as np

def plot_dataset_samples(dataset, dataset_name="Dataset", n_samples=5):
    """
    Plot random samples from a polarization dataset.
    Each sample shows 4 input channels + derived quantities (I, Q, U, DOP, AOP).

    Parameters:
    - dataset: torch Dataset returning (maps, vector, angles).
    - dataset_name (str): Title for the plot.
    - n_samples (int): Number of random samples to display.
    """
    fig, axes = plt.subplots(n_samples, 9, figsize=(18, 3 * n_samples))

    # Randomly pick samples
    sample_indexes = random.sample(range(len(dataset)), n_samples)

    for i, idx in enumerate(sample_indexes):
        maps, vector, _ = dataset[idx]
        maps = maps.float().numpy()
        vector = vector.float().numpy()

        # Image center and arrow radius
        h, w = maps.shape[1], maps.shape[2]
        center = (w / 2, h / 2)
        radius = 0.45 * min(h, w)

        # Polarization channels
        for ch in range(4):
            axes[i, ch].imshow(maps[ch], cmap="gray")
            axes[i, ch].axis("off")
            if i == 0:
                axes[i, ch].set_title(f"CH {ch+1}")

        # Derived quantities
        I = 0.5 * maps.sum(axis=0)
        Q = maps[0] - maps[2]
        U = maps[1] - maps[3]
        DOP = np.sqrt(Q**2 + U**2) / (I + 1e-8)
        AOP = 0.5 * np.arctan2(U, Q)

        derived = [I, Q, U, DOP, AOP]
        derived_titles = ["I", "Q", "U", "DOP", "AOP"]
        derived_cmaps = ["gray", "gray", "gray", "gray", "hsv"]

        for j, (img, title, cmap) in enumerate(zip(derived, derived_titles, derived_cmaps)):
            col_idx = j + 4
            axes[i, col_idx].imshow(img, cmap=cmap)
            axes[i, col_idx].axis("off")
            if i == 0:
                axes[i, col_idx].set_title(title)

        # GT arrow
        for j in range(9):
            axes[i, j].arrow(
                center[0], center[1],
                radius * vector[0], -radius * vector[1],
                color="red", head_width=3,
            )

        # Label sample index
        axes[i, 0].text(
            -0.2, 0.5, f"Idx: {idx}",
            fontsize=10, ha="right", va="center",
            transform=axes[i, 0].transAxes, rotation=45,
        )

    plt.suptitle(f"{dataset_name} - random {n_samples} samples", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_raw_vs_augmented_angles(dataset_raw, dataset_aug, train_indices, val_indices, num_samples=3):
    """
    Plot raw vs augmented angle distributions (polar histograms).

    Parameters:
    - dataset_raw: Dataset without augmentation (must have .angles).
    - dataset_aug: Dataset with augmentation.
    - train_indices, val_indices: Index tensors for train/validation split.
    - num_samples (int): Number of augmentation sampling passes.
    """

    def collect_augmented_angles(dataset, indices, num_samples=3):
        angles = []
        for _ in range(num_samples):
            for idx in indices:
                _, _, angle = dataset[idx]
                angles.append(angle.item())
        return torch.tensor(angles)

    # Collect data
    angles_raw_train = dataset_raw.angles[train_indices]
    angles_raw_val   = dataset_raw.angles[val_indices]
    angles_aug_train = collect_augmented_angles(dataset_aug, train_indices, num_samples)
    angles_aug_val   = collect_augmented_angles(dataset_aug, val_indices, num_samples)

    # Plot
    fig, axs = plt.subplots(2, 2, subplot_kw={"projection": "polar"}, figsize=(12, 8))

    # Raw
    axs[0,0].hist(angles_raw_train / 180 * 3.14, bins=100)
    axs[0,0].set_title("Train (Raw)")
    axs[0,1].hist(angles_raw_val / 180 * 3.14, bins=100)
    axs[0,1].set_title("Validation (Raw)")

    # Augmented
    axs[1,0].hist(angles_aug_train / 180 * 3.14, bins=100)
    axs[1,0].set_title("Train (Augmented)")
    axs[1,1].hist(angles_aug_val / 180 * 3.14, bins=100)
    axs[1,1].set_title("Validation (Augmented)")

    # Polar formatting
    for ax in axs.ravel():
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

    fig.suptitle("Distribution of azimuth angles (Raw vs Augmented)")
    plt.subplots_adjust(hspace=0.3, wspace=0.4)
    plt.show()
    return fig


def plot_angle_histogram(pred, gt):
    """
    Plots a histogram of angles in polar coordinates.

    Parameters:
    - pred (array-like): Predicted angles in degrees.
    - gt (array-like): Ground truth angles in degrees.
    """
    fig, ax = plt.subplots(1, subplot_kw={"projection": "polar"})
    for data, data_type in zip([pred, gt], ["Prediction", "Ground Truth"]):
        ax.hist(data / 180 * 3.14, bins=100, label=data_type, alpha=0.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig


def plot_error_histograms(train_error_tensor, val_error_tensor):
    """
    Plot histogram of training and validation angle errors.
    """
    train_errors = train_error_tensor.cpu().numpy()
    val_errors   = val_error_tensor.cpu().numpy()

    plt.figure(figsize=(8,5))
    plt.hist(train_errors, bins=50, alpha=0.5, label="Train")
    plt.hist(val_errors, bins=50, alpha=0.5, label="Validation")
    plt.xlabel("Angle Error [deg]")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")
    plt.legend()
    plt.show()


def plot_predictions_scaled(dataset, model, device, n_samples=5):
    """
    Plot sample images with GT and predicted vectors as arrows (scaled).

    Parameters:
    - dataset: Dataset that returns (maps, vector_gt, angle_gt).
    - model: Trained model with .get_angle() method.
    - device: Torch device to run inference on.
    - n_samples (int): Number of random samples to display.
    """
    model.eval()
    sample_indices = random.sample(range(len(dataset)), n_samples)

    for idx in sample_indices:
        maps, vector_gt, angle_gt = dataset[idx]

        maps_tensor = maps.unsqueeze(0).to(device)
        vector_pred = model(maps_tensor)
        angle_pred = model.get_angle(vector_pred)[0].item()

        height, width = maps.shape[1], maps.shape[2]
        start_pos = width // 2, height // 2
        max_length = min(height, width) * 0.4  # max arrow length

        # Ground truth arrow
        gt_length = (vector_gt[0]**2 + vector_gt[1]**2)**0.5
        scale_gt = max_length / max(gt_length.item(), 1e-6)
        dx_gt, dy_gt = vector_gt[0].item() * scale_gt, -vector_gt[1].item() * scale_gt

        # Prediction arrow
        pred_vec = vector_pred[0]
        pred_length = (pred_vec[0]**2 + pred_vec[1]**2)**0.5
        scale_pred = max_length / max(pred_length.item(), 1e-6)
        dx_pred, dy_pred = pred_vec[0].item() * scale_pred, -pred_vec[1].item() * scale_pred

        # Plot channels
        fig, axes = plt.subplots(1, maps.shape[0], figsize=(12, 3))
        if maps.shape[0] == 1:
            axes = [axes]  # Ensure iterable for single-channel
        for ch, ax in enumerate(axes):
            ax.imshow(maps[ch], cmap="gray")
            ax.axis("off")
            ax.arrow(start_pos[0], start_pos[1], dx_gt, dy_gt,
                     color="red", head_width=max_length*0.05, label="GT")
            ax.arrow(start_pos[0], start_pos[1], dx_pred, dy_pred,
                     color="blue", head_width=max_length*0.05, label="Pred")
            ax.legend(loc="upper right")

        plt.suptitle(f"GT Angle: {angle_gt:.2f}°, Pred Angle: {angle_pred:.2f}°")
        plt.show()
