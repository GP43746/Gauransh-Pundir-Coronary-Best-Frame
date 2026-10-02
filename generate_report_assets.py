"""
Temporary report-figure/table generator for the coronary-best-frame project.

Run from anywhere:
    python generate_report_assets.py

It uses the project code/config already present in the repository and the development
dataset paths used by the evaluation script. All temporary report assets are written to:

    <project_root>/outputs/table/

Generated assets:
    Figure 1  pipeline_diagram.png
    Figure 2  preprocessing_vesselness_mask.png
    Figure 3  ground_truth_rank_by_case.png
    Figure 4  successful_score_curve_case_dev_008.png
    Figure 5  failure_case_dev_003.png
    Figure 6  success_vs_failure_comparison.png
    Table 1   table_1_feature_selection_summary.csv / .png
    Table 2   table_2_aggregate_metrics.csv / .png
    Table 3   table_3_per_case_results.csv / .png

This is intentionally temporary. Delete this script and outputs/table/ after inserting
the assets into the report.
"""

from pathlib import Path
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import cv2

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(r"E:\OneDrive\Desktop\RNT Health Insights\coronary-best-frame")
DATA_ROOT = Path(r"E:\OneDrive\Desktop\Learn\Computer Vision\angiogram-keyframe\data\dev")
OUT = PROJECT_ROOT / "outputs" / "table"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import process_window  # noqa: E402


# ---------------------------------------------------------------------
# Locked evaluation data from the completed development evaluation
# ---------------------------------------------------------------------
PER_CASE = pd.DataFrame([
    ["case_dev_001", 28, 47, 45, 6, 0.814815, 0, 0, 0],
    ["case_dev_002", 33, 99, 104, 4, 0.906250, 0, 0, 1],
    ["case_dev_003", 19, 32, 26, 13, 0.333333, 0, 0, 0],
    ["case_dev_004", 17, 38, 31, 9, 0.500000, 0, 0, 0],
    ["case_dev_005", 11, 23, 26, 3, 0.800000, 0, 1, 1],
    ["case_dev_006", 25, 29, 29, 1, 1.000000, 1, 1, 1],
    ["case_dev_007", 15, 16, 17, 3, 0.857143, 0, 1, 1],
    ["case_dev_008", 16, 25, 25, 1, 1.000000, 1, 1, 1],
    ["case_dev_009", 18, 17, 19, 3, 0.882353, 0, 1, 1],
    ["case_dev_010", 16, 18, 18, 1, 1.000000, 1, 1, 1],
], columns=[
    "case_id", "num_frames", "ground_truth_frame", "predicted_frame",
    "ground_truth_rank", "normalized_rank", "top1", "top3", "top5"
])

AGGREGATE = pd.DataFrame([
    ["Number of cases", 10],
    ["Total candidate frames", 198],
    ["Mean ground-truth rank", 4.4],
    ["Median ground-truth rank", 3.0],
    ["Mean normalized rank", 0.8094],
    ["Top-1 accuracy", "30%"],
    ["Top-3 accuracy", "60%"],
    ["Top-5 accuracy", "70%"],
], columns=["Metric", "Value"])

FEATURE_SELECTION = pd.DataFrame([
    ["Vessel", "largest_component_fraction", "Retained", "Measures spatial extent of connected vessel response."],
    ["Vessel", "vessel_gradient_energy", "Retained", "Measures gradient strength within detected vessel regions."],
    ["Vessel", "local_vessel_background_contrast", "Retained", "Measures vessel-to-local-background separation."],
    ["Vessel", "skeleton_continuity", "Rejected", "Highly redundant with largest-component fraction (rho ≈ 0.99); did not improve combinations."],
    ["Quality", "tenengrad", "Retained", "Global gradient-based sharpness measure."],
    ["Quality", "local_rms_contrast", "Retained", "Captures local intensity variation rather than only global contrast."],
    ["Quality", "high-frequency energy", "Rejected", "Investigated but weaker than the retained quality pair."],
    ["Quality", "global gradient energy", "Rejected", "Investigated but weaker/redundant relative to retained quality features."],
    ["Quality", "laplacian variance", "Rejected", "Investigated but weaker than retained quality features."],
    ["Artifact", "blackhat_energy", "Rejected", "Only promising artifact candidate, but not a sufficiently validated proxy for actual artifact burden."],
], columns=["Branch", "Feature", "Decision", "Reason"])

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def frame_files(case_id):
    case_dir = DATA_ROOT / case_id
    files = sorted(
        [p for p in case_dir.iterdir()
         if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}],
        key=lambda p: p.name
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {case_dir}")
    return files


def load_frames(case_id):
    paths = frame_files(case_id)
    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read {p}")
        frames.append(img)
    return paths, frames


def save_table_png(df, path, title, font_size=9):
    fig_h = max(2.2, 0.42 * (len(df) + 1) + 0.7)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.45)
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def get_case_result(case_id):
    paths, frames = load_frames(case_id)
    result = process_window(frames)
    return paths, frames, result


def score_values(result):
    return np.asarray(
        [score["final_score"] for score in result["scores"]],
        dtype=float,
    )


# ---------------------------------------------------------------------
# Figure 1: Pipeline diagram
# ---------------------------------------------------------------------
def make_figure_1():
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        (0.4, 1.55, 1.65, 1.25, "Input\nFrame Window"),
        (2.35, 1.55, 1.65, 1.25, "P1–P99\nNormalization"),
        (4.30, 1.55, 1.65, 1.25, "Sato\nVesselness"),
        (6.25, 1.55, 1.65, 1.25, "Vessel Mask\nP90 + Cleanup"),
        (8.20, 1.55, 1.65, 1.25, "Feature\nExtraction"),
        (10.15, 1.55, 1.65, 1.25, "P5–P95\nNormalization"),
        (12.10, 2.30, 1.65, 1.25, "V / Q / T\nBranch Scores"),
        (12.10, 0.55, 1.65, 1.25, "0.40V + 0.40Q\n+ 0.20T"),
        (13.05, 3.25, 1.65, 0.75, "Best Frame"),
    ]

    for x, y, w, h, label in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.5,
            fill=False,
        )
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=9, fontweight="bold")

    arrows = [
        ((2.05, 2.175), (2.35, 2.175)),
        ((4.00, 2.175), (4.30, 2.175)),
        ((5.95, 2.175), (6.25, 2.175)),
        ((7.90, 2.175), (8.20, 2.175)),
        ((9.85, 2.175), (10.15, 2.175)),
        ((11.80, 2.175), (12.10, 2.925)),
        ((12.925, 2.30), (12.925, 1.80)),
        ((13.75, 1.80), (13.75, 3.25)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=14, linewidth=1.4
        ))

    ax.text(
        8.9, 0.25,
        "Vessel: LCF + vessel gradient energy + local vessel/background contrast\n"
        "Quality: Tenengrad + local RMS contrast\n"
        "Temporal: previous-frame and window-peak ratios",
        ha="center", va="center", fontsize=8.5
    )

    ax.set_title("Proposed Classical Computer-Vision Best-Frame Selection Pipeline",
                 fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_1_pipeline_diagram.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2: preprocessing + vesselness + mask
# ---------------------------------------------------------------------
def make_figure_2():
    paths, frames, result = get_case_result("case_dev_008")

    idx = int(result["best_index"])
    raw = frames[idx]
    pre = result["preprocessed_frames"][idx]
    vesselness = result["vesselness_maps"][idx]
    mask = result["vessel_masks"][idx]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title(f"(a) Original\n{paths[idx].name}")

    axes[1].imshow(pre, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("(b) P1–P99 normalized")

    axes[2].imshow(vesselness, cmap="gray")
    axes[2].set_title("(c) Sato vesselness\nσ = [1, 2, 3]")

    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("(d) Final vessel mask\nP90 + CC + closing")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Preprocessing and Vessel Representation Example: case_dev_008",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "Figure_2_preprocessing_vesselness_mask.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3: GT rank by case
# ---------------------------------------------------------------------
def make_figure_3():
    d = PER_CASE.copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(d["case_id"], d["ground_truth_rank"])
    ax.set_ylabel("Ground-truth rank")
    ax.set_xlabel("Development case")
    ax.set_title("Ground-Truth Frame Rank Across Development Cases",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(d["ground_truth_rank"]) + 2)
    ax.tick_params(axis="x", rotation=45)

    for bar, val in zip(bars, d["ground_truth_rank"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                str(int(val)), ha="center", va="bottom", fontsize=9)

    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_3_ground_truth_rank_by_case.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 4: successful score curve, case 008
# ---------------------------------------------------------------------
def make_figure_4():
    paths, frames, result = get_case_result("case_dev_008")
    scores = score_values(result)
    gt_frame = int(PER_CASE.loc[PER_CASE.case_id == "case_dev_008",
                                "ground_truth_frame"].iloc[0])
    pred_idx = int(result["best_index"])
    pred_frame_name = paths[pred_idx].stem

    # Match GT by frame filename stem when possible; otherwise use metadata frame number.
    gt_idx = None
    for i, p in enumerate(paths):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits and int(digits) == gt_frame:
            gt_idx = i
            break

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(scores)) + 1
    ax.plot(x, scores, marker="o", linewidth=1.8, markersize=4, label="Final score")
    ax.axvline(pred_idx + 1, linestyle="--", linewidth=1.5,
               label=f"Selected: {pred_frame_name}")
    if gt_idx is not None:
        ax.axvline(gt_idx + 1, linestyle=":", linewidth=1.8,
                   label=f"Ground truth: frame {gt_frame}")

    ax.set_xlabel("Frame position in candidate window")
    ax.set_ylabel("Final normalized score")
    ax.set_title("Frame Scores for Successful Selection: case_dev_008",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "Figure_4_successful_score_curve_case_dev_008.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 5: case 003 failure visualization
# ---------------------------------------------------------------------
def make_figure_5():
    paths, frames, result = get_case_result("case_dev_003")
    scores = score_values(result)
    gt_frame = 32
    pred_idx = int(result["best_index"])

    gt_idx = None
    for i, p in enumerate(paths):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits and int(digits) == gt_frame:
            gt_idx = i
            break

    if gt_idx is None:
        gt_idx = int(np.argmin(np.abs(np.arange(len(frames)) - 0)))

    candidate_indices = [pred_idx, gt_idx]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for col, i in enumerate(candidate_indices):
        axes[0, col].imshow(frames[i], cmap="gray")
        label = "Predicted" if i == pred_idx else "Ground truth"
        axes[0, col].set_title(
            f"{label}\n{paths[i].name}\nscore={scores[i]:.4f}"
        )
        axes[0, col].axis("off")

    axes[1, 0].plot(np.arange(len(scores)) + 1, scores, marker="o", ms=3)
    axes[1, 0].axvline(pred_idx + 1, linestyle="--", linewidth=1.5)
    axes[1, 0].axvline(gt_idx + 1, linestyle=":", linewidth=1.5)
    axes[1, 0].set_xlabel("Frame position")
    axes[1, 0].set_ylabel("Final score")
    axes[1, 0].set_title("Score profile")
    axes[1, 0].grid(alpha=0.25)

    # Show vessel mask for the selected prediction as an interpretability panel.
    axes[1, 1].imshow(result["vessel_masks"][pred_idx], cmap="gray")
    axes[1, 1].set_title("Vessel mask of predicted frame")
    axes[1, 1].axis("off")

    fig.suptitle(
        "Failure Analysis: case_dev_003\n"
        "Ground truth = frame 32; predicted frame = 26; GT rank = 13/19",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "Figure_5_failure_case_dev_003.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 6: successful vs failure comparison
# ---------------------------------------------------------------------
def make_figure_6():
    cases = ["case_dev_008", "case_dev_003"]
    titles = ["Successful case: case_dev_008", "Failure case: case_dev_003"]

    results = {}
    for case in cases:
        paths, frames, result = get_case_result(case)
        results[case] = (paths, frames, result)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Row 1: success
    paths, frames, result = results["case_dev_008"]
    scores = score_values(result)
    pred = int(result["best_index"])
    axes[0, 0].imshow(frames[pred], cmap="gray")
    axes[0, 0].set_title(f"Selected frame\n{paths[pred].name}")
    axes[0, 1].imshow(result["vesselness_maps"][pred], cmap="gray")
    axes[0, 1].set_title("Sato vesselness")
    axes[0, 2].plot(np.arange(len(scores)) + 1, scores, marker="o", ms=3)
    axes[0, 2].axvline(pred + 1, linestyle="--")
    axes[0, 2].set_title("Score profile")

    # Row 2: failure
    paths, frames, result = results["case_dev_003"]
    scores = score_values(result)
    pred = int(result["best_index"])
    gt = None
    for i, p in enumerate(paths):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        if digits and int(digits) == 32:
            gt = i
            break
    if gt is None:
        gt = 0

    axes[1, 0].imshow(frames[pred], cmap="gray")
    axes[1, 0].set_title(f"Selected frame\n{paths[pred].name}")
    axes[1, 1].imshow(frames[gt], cmap="gray")
    axes[1, 1].set_title(f"Ground truth\n{paths[gt].name}")
    axes[1, 2].plot(np.arange(len(scores)) + 1, scores, marker="o", ms=3)
    axes[1, 2].axvline(pred + 1, linestyle="--", label="Prediction")
    axes[1, 2].axvline(gt + 1, linestyle=":", label="Ground truth")
    axes[1, 2].set_title("Score profile")
    axes[1, 2].legend()

    for row in axes:
        for ax in row:
            ax.axis("off") if ax in row[:2] else None
            if ax in row[:2]:
                ax.set_xticks([])
                ax.set_yticks([])

    # Restore score-chart axes because the loop above deliberately only removes them from image panels.
    axes[0, 2].set_xlabel("Frame position")
    axes[0, 2].set_ylabel("Score")
    axes[0, 2].grid(alpha=0.2)
    axes[1, 2].set_xlabel("Frame position")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].grid(alpha=0.2)

    fig.suptitle(
        "Comparison of a Successful Selection and a Failure Case",
        fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT / "Figure_6_success_vs_failure_comparison.png",
                dpi=240, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def make_tables():
    # CSV files
    FEATURE_SELECTION.to_csv(OUT / "Table_1_feature_selection_summary.csv", index=False)
    AGGREGATE.to_csv(OUT / "Table_2_aggregate_metrics.csv", index=False)
    PER_CASE.to_csv(OUT / "Table_3_per_case_results.csv", index=False)

    # Pretty PNGs
    save_table_png(
        FEATURE_SELECTION,
        OUT / "Table_1_feature_selection_summary.png",
        "Table 1. Feature Selection Summary",
        font_size=8
    )
    save_table_png(
        AGGREGATE,
        OUT / "Table_2_aggregate_metrics.png",
        "Table 2. Development-Set Aggregate Metrics",
        font_size=9
    )
    save_table_png(
        PER_CASE,
        OUT / "Table_3_per_case_results.png",
        "Table 3. Per-Case Development Results",
        font_size=8
    )


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root:    {DATA_ROOT}")
    print(f"Output dir:   {OUT}")

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Project root does not exist: {PROJECT_ROOT}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root does not exist: {DATA_ROOT}")

    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()
    make_figure_5()
    make_figure_6()
    make_tables()

    print("\nGenerated assets:")
    for p in sorted(OUT.iterdir()):
        print(" -", p.name)

    print("\nDone. These are temporary report assets.")
    print("Delete generate_report_assets.py and outputs/table/ after inserting them into the report.")


if __name__ == "__main__":
    main()
