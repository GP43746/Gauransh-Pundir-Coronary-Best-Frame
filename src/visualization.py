from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def _to_display_image(frame):
    """
    Convert an input frame into a displayable grayscale image.
    """
    if isinstance(frame, (str, Path)):
        image = cv2.imread(
            str(frame),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {frame}"
            )
    elif isinstance(frame, np.ndarray):
        image = frame
    else:
        raise TypeError(
            "Frame must be a path or numpy.ndarray."
        )

    if image.ndim == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    return image


def plot_score_grid(
    frames,
    scores,
    save_path=None,
    frame_labels=None,
    gt_index=None,
    title="Best-Frame Selection",
):
    """
    Create a grid showing every candidate frame and its score.

    Parameters
    ----------
    frames : sequence
        Candidate frames.
    scores : sequence[dict]
        Score dictionaries produced by the pipeline.
    save_path : str | Path | None
        Output path. If None, the figure is not saved.
    frame_labels : sequence[str] | None
        Optional labels for frames.
    gt_index : int | None
        Optional ground-truth frame index for development
        evaluation.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """
    if len(frames) != len(scores):
        raise ValueError(
            "frames and scores must have the same length."
        )

    n = len(frames)

    if n == 0:
        raise ValueError("frames cannot be empty.")

    if frame_labels is None:
        frame_labels = [
            f"Frame {i + 1}"
            for i in range(n)
        ]

    if len(frame_labels) != n:
        raise ValueError(
            "frame_labels must match the number of frames."
        )

    best_index = int(
        np.argmax(
            [
                score["final_score"]
                for score in scores
            ]
        )
    )

    columns = min(5, n)
    rows = int(np.ceil(n / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.2 * rows),
    )

    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        image = _to_display_image(frames[i])

        axes[i].imshow(
            image,
            cmap="gray",
        )

        score = scores[i]["final_score"]

        axes[i].set_title(
            f"{frame_labels[i]}\n"
            f"Score: {score:.4f}"
        )

        axes[i].axis("off")

        if i == best_index:
            for spine in axes[i].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)

        if gt_index is not None and i == gt_index:
            axes[i].set_xlabel(
                "GROUND TRUTH",
                fontweight="bold",
            )

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    return fig


def plot_score_curve(
    scores,
    save_path=None,
    gt_index=None,
    title="Frame Score Across Candidate Window",
):
    """
    Plot final frame scores across the candidate window.

    Parameters
    ----------
    scores : sequence[dict]
        Score dictionaries produced by the pipeline.
    save_path : str | Path | None
        Output path.
    gt_index : int | None
        Optional ground-truth index.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not scores:
        raise ValueError("scores cannot be empty.")

    final_scores = np.array(
        [
            score["final_score"]
            for score in scores
        ],
        dtype=np.float64,
    )

    frame_indices = np.arange(
        len(final_scores)
    )

    best_index = int(
        np.argmax(final_scores)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        frame_indices,
        final_scores,
        marker="o",
        linewidth=1.8,
    )

    ax.scatter(
        [best_index],
        [final_scores[best_index]],
        s=100,
        zorder=5,
    )

    if gt_index is not None:
        ax.scatter(
            [gt_index],
            [final_scores[gt_index]],
            marker="x",
            s=120,
            linewidths=2,
            zorder=6,
        )

    ax.set_xlabel(
        "Frame index within candidate window"
    )

    ax.set_ylabel(
        "Final normalized score"
    )

    ax.set_title(title)

    ax.set_xticks(frame_indices)

    ax.grid(
        alpha=0.25,
        linestyle="--",
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    return fig


def plot_interpretability(
    frame,
    vesselness,
    vessel_mask,
    score,
    save_path=None,
    title="Frame Interpretation",
):
    """
    Visualize the intermediate representations for one frame.

    Panels:
        1. Original frame
        2. Sato vesselness
        3. Final vessel mask
        4. Branch/final scores
    """
    image = _to_display_image(frame)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        image,
        cmap="gray",
    )

    axes[0].set_title(
        "Input Frame"
    )

    axes[1].imshow(
        vesselness,
        cmap="gray",
    )

    axes[1].set_title(
        "Sato Vesselness"
    )

    axes[2].imshow(
        vessel_mask,
        cmap="gray",
    )

    axes[2].set_title(
        "Vessel Mask"
    )

    axes[3].axis("off")

    score_text = (
        f"Vessel:   {score['vessel_score']:.4f}\n"
        f"Quality:  {score['quality_score']:.4f}\n"
        f"Temporal: {score['temporal_score']:.4f}\n"
        f"\n"
        f"Final:    {score['final_score']:.4f}"
    )

    axes[3].text(
        0.05,
        0.5,
        score_text,
        fontsize=13,
        verticalalignment="center",
        family="monospace",
    )

    for ax in axes[:3]:
        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    return fig
