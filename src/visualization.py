from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def _to_display_image(image):
    """
    Convert an image to a displayable [0, 1] grayscale image.
    """
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            arr = (
                0.299 * arr[..., 0]
                + 0.587 * arr[..., 1]
                + 0.114 * arr[..., 2]
            )
        else:
            arr = arr[..., 0]

    finite = np.isfinite(arr)

    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    values = arr[finite]

    low = np.percentile(values, 1)
    high = np.percentile(values, 99)

    if high <= low:
        low = values.min()
        high = values.max()

    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)

    arr = (arr - low) / (high - low)
    return np.clip(arr, 0.0, 1.0)


def _save_figure(fig, save_path):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)
    else:
        plt.show()


def _frame_label(frame_id):
    return f"frame_{int(frame_id):03d}"


# ---------------------------------------------------------------------
# 1. Frame-score montage
# ---------------------------------------------------------------------

def plot_score_grid(
    frames,
    scores,
    save_path=None,
    frame_labels=None,
    gt_index=None,
    selected_index=None,
    title="Frame scores",
    max_columns=5,
):
    """
    Display candidate frames with their final scores.

    Parameters
    ----------
    frames : list
        Images.
    scores : list
        Final frame scores.
    frame_labels : list, optional
        Labels such as frame_043.
    gt_index : int, optional
        Ground-truth frame position.
    selected_index : int, optional
        Selected frame position.
    """

    n = len(frames)

    if n == 0:
        return

    columns = min(max_columns, n)
    rows = int(np.ceil(n / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 3.2 * rows),
    )

    axes = np.atleast_1d(axes).ravel()

    scores = np.asarray(scores)

    for i in range(n):

        ax = axes[i]

        ax.imshow(
            _to_display_image(frames[i]),
            cmap="gray",
        )

        label = (
            frame_labels[i]
            if frame_labels is not None
            else f"Frame {i}"
        )

        ax.set_title(
            f"{label}\nScore: {scores[i]:.3f}",
            fontsize=9,
        )

        if gt_index is not None and i == gt_index:
            ax.set_title(
                f"{label}\nGT | Score: {scores[i]:.3f}",
                fontsize=9,
                fontweight="bold",
            )

        if selected_index is not None and i == selected_index:
            ax.set_title(
                f"{label}\nSELECTED | Score: {scores[i]:.3f}",
                fontsize=9,
                fontweight="bold",
            )

        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 2. Final score curve
# ---------------------------------------------------------------------

def plot_score_curve(
    scores,
    save_path=None,
    frame_ids=None,
    gt_index=None,
    selected_index=None,
    title="Final frame score",
):
    """
    Plot final score against frame position.
    """

    scores = np.asarray(scores, dtype=float)

    if frame_ids is None:
        frame_ids = np.arange(len(scores))

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        frame_ids,
        scores,
        marker="o",
        linewidth=2,
    )

    if gt_index is not None:
        ax.axvline(
            frame_ids[gt_index],
            linestyle="--",
            linewidth=1.5,
            label="Ground truth",
        )

    if selected_index is not None:
        ax.axvline(
            frame_ids[selected_index],
            linestyle=":",
            linewidth=2,
            label="Selected",
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Final score")
    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 3. Branch-score curves
# ---------------------------------------------------------------------

def plot_branch_scores(
    frame_results,
    frame_ids,
    save_path=None,
    gt_index=None,
    selected_index=None,
    title="Scoring branches",
):
    """
    Plot vessel, quality, temporal and final scores.
    """

    vessel = [
        r.get("vessel_score", np.nan)
        for r in frame_results
    ]

    quality = [
        r.get("quality_score", np.nan)
        for r in frame_results
    ]

    temporal = [
        r.get("temporal_score", np.nan)
        for r in frame_results
    ]

    final = [
        r.get("final_score", np.nan)
        for r in frame_results
    ]

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        frame_ids,
        vessel,
        marker="o",
        label="Vessel branch",
    )

    ax.plot(
        frame_ids,
        quality,
        marker="o",
        label="Quality branch",
    )

    ax.plot(
        frame_ids,
        temporal,
        marker="o",
        label="Temporal branch",
    )

    ax.plot(
        frame_ids,
        final,
        marker="o",
        linewidth=3,
        label="Final score",
    )

    if gt_index is not None:
        ax.axvline(
            frame_ids[gt_index],
            linestyle="--",
            linewidth=1.5,
            label="Ground truth",
        )

    if selected_index is not None:
        ax.axvline(
            frame_ids[selected_index],
            linestyle=":",
            linewidth=2,
            label="Selected",
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(-0.05, 1.05)

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 4. Feature heatmap
# ---------------------------------------------------------------------

def plot_feature_heatmap(
    frame_results,
    frame_ids,
    save_path=None,
    title="Normalized feature profile",
):
    """
    Visualize all normalized features across the candidate window.
    """

    excluded = {
        "frame_index",
        "frame_id",
        "vessel_score",
        "quality_score",
        "temporal_score",
        "final_score",
    }

    feature_names = []

    for result in frame_results:

        for key, value in result.items():

            if not key.endswith("_norm"):
                continue

            if key in excluded:
                continue

            if key not in feature_names:
                feature_names.append(key)

    if not feature_names:
        return

    matrix = np.full(
        (len(feature_names), len(frame_results)),
        np.nan,
        dtype=float,
    )

    for row, feature in enumerate(feature_names):

        for col, result in enumerate(frame_results):

            value = result.get(feature, np.nan)

            try:
                matrix[row, col] = float(value)
            except (TypeError, ValueError):
                matrix[row, col] = np.nan

    fig, ax = plt.subplots(
        figsize=(
            max(10, len(frame_ids) * 0.45),
            max(5, len(feature_names) * 0.45),
        )
    )

    masked = np.ma.masked_invalid(matrix)

    image = ax.imshow(
        masked,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )

    ax.set_xticks(
        np.arange(len(frame_ids))
    )

    ax.set_xticklabels(
        [str(x) for x in frame_ids],
        rotation=90,
        fontsize=8,
    )

    ax.set_yticks(
        np.arange(len(feature_names))
    )

    ax.set_yticklabels(
        [
            feature.replace("_norm", "")
            .replace("_", " ")
            for feature in feature_names
        ],
        fontsize=9,
    )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Feature")

    ax.set_title(
        title,
        fontweight="bold",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Normalized value"
    )

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 5. Temporal feature curves
# ---------------------------------------------------------------------

def plot_temporal_features(
    frame_results,
    frame_ids,
    save_path=None,
    title="Temporal feature behavior",
):
    """
    Plot all normalized temporal features.
    """

    temporal_features = [
        "sharpness_ratio_prev_norm",
        "contrast_ratio_prev_norm",
        "sharpness_peak_ratio_norm",
        "vessel_gradient_peak_ratio_norm",
        "vessel_contrast_peak_ratio_norm",
    ]

    available = [
        feature
        for feature in temporal_features
        if any(
            feature in result
            for result in frame_results
        )
    ]

    if not available:
        return

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    for feature in available:

        values = [
            result.get(feature, np.nan)
            for result in frame_results
        ]

        ax.plot(
            frame_ids,
            values,
            marker="o",
            label=(
                feature
                .replace("_norm", "")
                .replace("_", " ")
            ),
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Normalized value")
    ax.set_ylim(-0.05, 1.05)

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.grid(alpha=0.25)
    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 6. Branch contribution plot
# ---------------------------------------------------------------------

def plot_branch_contributions(
    frame_results,
    frame_ids,
    vessel_weight=0.40,
    quality_weight=0.40,
    temporal_weight=0.20,
    save_path=None,
    gt_index=None,
    selected_index=None,
    title="Weighted score contributions",
):
    """
    Show how each branch contributes to the final score.
    """

    vessel = np.asarray([
        r.get("vessel_score", np.nan)
        for r in frame_results
    ])

    quality = np.asarray([
        r.get("quality_score", np.nan)
        for r in frame_results
    ])

    temporal = np.asarray([
        r.get("temporal_score", np.nan)
        for r in frame_results
    ])

    vessel_contribution = vessel * vessel_weight
    quality_contribution = quality * quality_weight
    temporal_contribution = temporal * temporal_weight

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.stackplot(
        frame_ids,
        vessel_contribution,
        quality_contribution,
        temporal_contribution,
        labels=[
            f"Vessel × {vessel_weight:.2f}",
            f"Quality × {quality_weight:.2f}",
            f"Temporal × {temporal_weight:.2f}",
        ],
        alpha=0.8,
    )

    if gt_index is not None:
        ax.axvline(
            frame_ids[gt_index],
            linestyle="--",
            linewidth=1.5,
            label="Ground truth",
        )

    if selected_index is not None:
        ax.axvline(
            frame_ids[selected_index],
            linestyle=":",
            linewidth=2,
            label="Selected",
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Contribution to final score")

    ax.set_title(
        title,
        fontweight="bold",
    )

    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 7. Selected-frame interpretability
# ---------------------------------------------------------------------

def plot_interpretability(
    frame,
    vesselness,
    vessel_mask,
    score,
    save_path=None,
    title="Selected-frame interpretability",
):
    """
    Display:
        1. frame
        2. Sato vesselness
        3. vessel mask
        4. vesselness overlay
    """

    display_frame = _to_display_image(frame)
    display_vesselness = _to_display_image(vesselness)

    mask = np.asarray(
        vessel_mask,
        dtype=bool,
    )

    overlay = np.stack(
        [
            display_frame,
            display_frame,
            display_frame,
        ],
        axis=-1,
    )

    overlay[mask, 0] = 1.0
    overlay[mask, 1] *= 0.35
    overlay[mask, 2] *= 0.35

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 9),
    )

    axes[0, 0].imshow(
        display_frame,
        cmap="gray",
    )

    axes[0, 0].set_title(
        "Preprocessed frame"
    )

    axes[0, 1].imshow(
        display_vesselness,
        cmap="magma",
    )

    axes[0, 1].set_title(
        "Sato vesselness"
    )

    axes[1, 0].imshow(
        mask,
        cmap="gray",
    )

    axes[1, 0].set_title(
        "Vessel mask"
    )

    axes[1, 1].imshow(
        overlay
    )

    axes[1, 1].set_title(
        "Vessel-mask overlay"
    )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"{title}\nFinal score = {score:.4f}",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 8. Top-K frame comparison
# ---------------------------------------------------------------------

def plot_top_k_frames(
    frames,
    scores,
    frame_ids,
    k=5,
    gt_frame=None,
    save_path=None,
    title="Top-ranked candidate frames",
):
    """
    Display the highest-ranked candidate frames.
    """

    scores = np.asarray(scores)

    ranking = np.argsort(
        scores
    )[::-1]

    ranking = ranking[:min(k, len(ranking))]

    columns = len(ranking)

    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(3.2 * columns, 4),
    )

    axes = np.atleast_1d(axes)

    for rank, index in enumerate(ranking, start=1):

        ax = axes[rank - 1]

        ax.imshow(
            _to_display_image(frames[index]),
            cmap="gray",
        )

        frame_id = frame_ids[index]

        marker = ""

        if gt_frame is not None and frame_id == gt_frame:
            marker = " | GT"

        ax.set_title(
            f"Rank {rank}\n"
            f"frame_{frame_id:03d}\n"
            f"Score: {scores[index]:.4f}"
            f"{marker}",
            fontsize=9,
            fontweight=(
                "bold"
                if marker
                else "normal"
            ),
        )

        ax.axis("off")

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 9. Frame feature table visualization
# ---------------------------------------------------------------------

def plot_feature_profiles(
    frame_results,
    frame_ids,
    selected_index=None,
    gt_index=None,
    save_path=None,
    title="Feature profiles",
):
    """
    Plot selected important feature groups separately.
    """

    groups = {
        "Vessel features": [
            "largest_component_fraction_norm",
            "vessel_gradient_energy_norm",
            "local_vessel_background_contrast_norm",
        ],
        "Quality features": [
            "tenengrad_norm",
            "local_rms_contrast_norm",
        ],
    }

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
    )

    for ax, (group_name, features) in zip(
        axes,
        groups.items(),
    ):

        for feature in features:

            if not any(
                feature in result
                for result in frame_results
            ):
                continue

            values = [
                result.get(feature, np.nan)
                for result in frame_results
            ]

            ax.plot(
                frame_ids,
                values,
                marker="o",
                label=(
                    feature
                    .replace("_norm", "")
                    .replace("_", " ")
                ),
            )

        ax.set_ylabel("Normalized value")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            group_name,
            fontweight="bold",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    if gt_index is not None:
        axes[0].axvline(
            frame_ids[gt_index],
            linestyle="--",
            linewidth=1.5,
            label="Ground truth",
        )

        axes[1].axvline(
            frame_ids[gt_index],
            linestyle="--",
            linewidth=1.5,
        )

    if selected_index is not None:
        axes[0].axvline(
            frame_ids[selected_index],
            linestyle=":",
            linewidth=2,
            label="Selected",
        )

        axes[1].axvline(
            frame_ids[selected_index],
            linestyle=":",
            linewidth=2,
        )

    axes[-1].set_xlabel("Frame index")

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    _save_figure(fig, save_path)


# ---------------------------------------------------------------------
# 10. Complete case dashboard
# ---------------------------------------------------------------------

def plot_case_dashboard(
    frames,
    frame_ids,
    frame_results,
    vesselness,
    vessel_mask,
    best_index,
    gt_index=None,
    save_path=None,
    title="Case dashboard",
):
    """
    Compact research-style summary for one case.

    Contains:
        - selected frame
        - vesselness
        - mask
        - final score curve
        - branch score curves
        - top candidate comparison
    """

    scores = np.asarray([
        result["final_score"]
        for result in frame_results
    ])

    best_frame = frames[best_index]

    fig = plt.figure(
        figsize=(16, 11)
    )

    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[1, 1, 1],
    )

    # -------------------------------------------------------------
    # Selected frame
    # -------------------------------------------------------------

    ax = fig.add_subplot(
        grid[0, 0]
    )

    ax.imshow(
        _to_display_image(best_frame),
        cmap="gray",
    )

    ax.set_title(
        f"Selected frame\n"
        f"frame_{frame_ids[best_index]:03d}",
        fontweight="bold",
    )

    ax.axis("off")

    # -------------------------------------------------------------
    # Vesselness
    # -------------------------------------------------------------

    ax = fig.add_subplot(
        grid[0, 1]
    )

    ax.imshow(
        _to_display_image(vesselness[best_index]),
        cmap="magma",
    )

    ax.set_title(
        "Sato vesselness"
    )

    ax.axis("off")

    # -------------------------------------------------------------
    # Mask
    # -------------------------------------------------------------

    ax = fig.add_subplot(
        grid[0, 2]
    )

    ax.imshow(
        vessel_mask[best_index],
        cmap="gray",
    )

    ax.set_title(
        "Vessel mask"
    )

    ax.axis("off")

    # -------------------------------------------------------------
    # Score summary
    # -------------------------------------------------------------

    ax = fig.add_subplot(
        grid[0, 3]
    )

    ax.axis("off")

    selected = frame_results[best_index]

    summary = (
        f"FINAL SCORE\n"
        f"{selected['final_score']:.4f}\n\n"
        f"Vessel:   {selected.get('vessel_score', np.nan):.4f}\n"
        f"Quality:  {selected.get('quality_score', np.nan):.4f}\n"
        f"Temporal: {selected.get('temporal_score', np.nan):.4f}"
    )

    if gt_index is not None:

        gt_frame = frame_ids[gt_index]

        summary += (
            f"\n\nGT FRAME\n"
            f"frame_{gt_frame:03d}"
        )

    ax.text(
        0.5,
        0.5,
        summary,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )

    # -------------------------------------------------------------
    # Final score
    # -------------------------------------------------------------

    ax = fig.add_subplot(
        grid[1, :2]
    )

    ax.plot(
        frame_ids,
        scores,
        marker="o",
        linewidth=2,
    )

    ax.axvline(
        frame_ids[best_index],
        linestyle=":",
        linewidth=2,
        label="Selected",
    )

    if gt_index is not None:
        ax.axvline(
            frame_ids[gt_index],
            linestyle="--",
            linewidth=1.5,
            label="Ground truth",
        )

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Final score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Final ranking score")
    ax.grid(alpha=0.25)
    ax.legend()

    # -------------------------------------------------------------
    # Branch scores
    # -------------------------------------------------------------

    ax = fig.add_subplot(
        grid[1, 2:]
    )

    branch_names = [
        "Vessel",
        "Quality",
        "Temporal",
        "Final",
    ]

    branch_values = [
        selected.get("vessel_score", np.nan),
        selected.get("quality_score", np.nan),
        selected.get("temporal_score", np.nan),
        selected.get("final_score", np.nan),
    ]

    ax.bar(
        branch_names,
        branch_values,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(
        "Selected-frame score decomposition"
    )

    # -------------------------------------------------------------
    # Top frames
    # -------------------------------------------------------------

    ranking = np.argsort(
        scores
    )[::-1][:5]

    top_scores = scores[ranking]

    top_labels = [
        f"frame_{frame_ids[i]:03d}"
        for i in ranking
    ]

    ax = fig.add_subplot(
        grid[2, :2]
    )

    ax.barh(
        top_labels[::-1],
        top_scores[::-1],
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Final score")
    ax.set_title("Top-ranked frames")

    # -------------------------------------------------------------
    # Feature summary
    # -------------------------------------------------------------

    ax = fig.add_subplot(
        grid[2, 2:]
    )

    feature_names = [
        "largest_component_fraction_norm",
        "vessel_gradient_energy_norm",
        "local_vessel_background_contrast_norm",
        "tenengrad_norm",
        "local_rms_contrast_norm",
    ]

    available = [
        feature
        for feature in feature_names
        if feature in selected
    ]

    values = [
        selected[feature]
        for feature in available
    ]

    labels = [
        feature
        .replace("_norm", "")
        .replace("_", "\n")
        for feature in available
    ]

    ax.bar(
        labels,
        values,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalized value")
    ax.set_title(
        "Selected-frame feature profile"
    )

    fig.suptitle(
        title,
        fontsize=17,
        fontweight="bold",
    )

    fig.tight_layout()

    _save_figure(fig, save_path)