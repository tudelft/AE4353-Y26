import matplotlib.pyplot as plt
import numpy as np


def plot_data(x1, x2, t, title="Dataset"):
    """Scatter the two classes: red circles for t = 0, green crosses for t = 1."""
    plt.figure(figsize=(5, 5))
    plt.plot(x1[t == 0], x2[t == 0], 'ro', markersize=3, label='t = 0')
    plt.plot(x1[t == 1], x2[t == 1], 'g+', markersize=5, label='t = 1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_decision_boundary(x1, x2, t, w1, w2, bias=0.0, title="Decision boundary"):
    """Scatter the data and overlay the line where the perceptron switches
    class, i.e. where w1 * x1 + w2 * x2 + bias = 0."""
    plt.figure(figsize=(5, 5))
    plt.plot(x1[t == 0], x2[t == 0], 'ro', markersize=3, label='t = 0')
    plt.plot(x1[t == 1], x2[t == 1], 'g+', markersize=5, label='t = 1')

    if w2 != 0:
        line_x = np.linspace(x1.min(), x1.max(), 100)
        line_y = -(w1 * line_x + bias) / w2
        plt.plot(line_x, line_y, 'k--', linewidth=2, label='decision boundary')
        plt.ylim(x2.min(), x2.max())

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_loss_curve(losses, title="Training loss"):
    """Plot the loss against the epoch number."""
    plt.figure(figsize=(6, 4))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean loss')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()
