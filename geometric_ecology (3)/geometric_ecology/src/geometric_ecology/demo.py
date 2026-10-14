"""Known synthetic mechanisms; group labels are used ONLY after mapping."""
import numpy as np
from .core import normalize


def synthetic_corpus(n=1100, dimensions=48, seed=42):
    if n < 240 or dimensions < 16:
        raise ValueError("Demo needs n>=240 and dimensions>=16.")
    rng = np.random.default_rng(seed)
    sizes = np.array([.21, .21, .075, .16, .13, .075, .10, .04])
    counts = (sizes * n).astype(int)
    counts[0] += n - counts.sum()
    pieces, labels = [], []
    names = ["basin A", "basin B", "narrow connector", "high-dimensional patch", "ring", "island", "redundant patch", "random outliers"]
    for group, count in zip(names, counts):
        z = np.zeros((count, dimensions))
        if group in ("basin A", "basin B", "redundant patch"):
            sign = -1 if group != "basin B" else 1
            angle = sign * .60
            z[:, 0], z[:, 1] = np.cos(angle), np.sin(angle)
            spread = .045 if group != "redundant patch" else .0015
            z[:, 1:4] += rng.normal(scale=spread, size=(count, 3))
        elif group == "narrow connector":
            theta = np.linspace(-.56, .56, count)
            z[:, 0], z[:, 1] = np.cos(theta), np.sin(theta)
            z[:, 2:4] = rng.normal(scale=.001, size=(count, 2))
        elif group == "high-dimensional patch":
            z[:, 0], z[:, 4] = .7, .7
            z[:, 5:15] = rng.normal(scale=.085, size=(count, 10))
        elif group == "ring":
            theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
            z[:, 0], z[:, 15] = .65, .65
            z[:, 6], z[:, 7] = .23 * np.cos(theta), .23 * np.sin(theta)
        elif group == "island":
            z[:, 0], z[:, 8] = -.6, .8
            z[:, 9:12] = rng.normal(scale=.03, size=(count, 3))
        else:
            z = rng.normal(size=(count, dimensions))
        pieces.append(z)
        labels.extend([group] * count)
    x = normalize(np.vstack(pieces))
    # Avoid making coordinate masking artificially align with planted features.
    rotation, _ = np.linalg.qr(rng.normal(size=(dimensions, dimensions)))
    x = normalize(x @ rotation)
    ids = [f"demo-{i:05d}" for i in range(n)]
    metadata = [{"title": f"Synthetic item {i}", "text": "A simulated embedding for geometric navigation. Mechanism labels appear only in the Evidence view.", "demo_group": label} for i, label in enumerate(labels)]
    return x, ids, metadata
