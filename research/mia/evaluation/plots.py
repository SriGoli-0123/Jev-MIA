"""Static diagnostic plots (PNG). Plotting is optional: failures are reported, never fatal."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mia.evaluation.metrics import roc_points  # noqa: E402


def roc_plot(curves, path, title):
    """``curves``: {label: (y, scores)}; log-scale FPR to show the low-FPR regime."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for label, (y, scores) in curves.items():
        fpr, tpr = roc_points(y, scores)
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, label=label)
    grid = np.logspace(-4, 0, 50)
    ax.plot(grid, grid, "k:", lw=0.8, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def membership_histograms(table, features, path, title):
    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 2.8), squeeze=False)
    for ax, feature in zip(axes[0], features):
        for label, name in ((1, "member"), (0, "non-member")):
            values = table.loc[table["member"] == label, feature].to_numpy(dtype=float)
            ax.hist(values[np.isfinite(values)], bins=40, alpha=0.5, density=True, label=name)
        ax.set_title(feature, fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def heatmap(matrix, rows, cols, path, title, fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    image = ax.imshow(matrix, vmin=0.3, vmax=0.8, cmap="viridis")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlabel("attack query primitive")
    ax.set_ylabel("training primitive")
    for r in range(len(rows)):
        for c in range(len(cols)):
            if np.isfinite(matrix[r][c]):
                ax.text(c, r, fmt.format(matrix[r][c]), ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(image, ax=ax, label="ROC-AUC")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def safe(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return None
    except Exception as exc:  # plotting must never invalidate a run
        return f"{type(exc).__name__}: {exc}"
