import numpy as np


def get_linear_data(w1=-2.0, w2=-2.0, threshold=0.0, n_samples=1000,
                    min_x=-10.0, max_x=10.0, seed=0):
    """Linearly separable two-dimensional classification data.

    Returns three flat arrays of shape (n_samples,): the two input features
    x1 and x2, and the binary target t.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(min_x, max_x, n_samples)
    x2 = rng.uniform(min_x, max_x, n_samples)
    t = ((w1 * x1 + w2 * x2) > threshold).astype(int)
    return x1, x2, t


def get_simple_data(n_samples=1000, seed=0):
    """A problem a single perceptron can solve: the two classes are separated
    by a straight line through the origin."""
    return get_linear_data(w1=-2.0, w2=-2.0, threshold=0.0,
                           n_samples=n_samples, seed=seed)


def get_difficult_data(n_samples=1000, seed=0):
    """An XOR-like problem. The two classes are *not* separable by a straight
    line, so a single perceptron cannot solve it no matter how it is trained."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-10.0, 10.0, n_samples)
    x2 = rng.uniform(-10.0, 10.0, n_samples)
    t = (np.sign(x1) * np.sign(x2) > 0).astype(int)
    return x1, x2, t
