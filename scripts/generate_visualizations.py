from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(
    r"E:\OneDrive\Desktop\RNT Health Insights\coronary-best-frame"
)

DATA_ROOT = Path(
    r"E:\OneDrive\Desktop\Learn\Computer Vision\angiogram-keyframe\data\dev"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "dev_visualizations"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.pipeline import process_window

from src.visualization import (
    plot_score_grid,
    plot_score_curve,
    plot_branch_scores,
    plot_feature_heatmap,
    plot_temporal_features,
    plot_branch_contributions,
    plot_interpretability,
    plot_top_k_frames,
    plot_feature_profiles,
    plot_case_dashboard,
)


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def frame_number(path):
    match = re.search(
        r"frame[_-]?(\d+)",
        path.stem.lower(),
    )

    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not extract frame number from {path}"
    )


def load_case_frames(case_dir):
    paths = sorted(
        [
            p
            for p in case_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=frame_number,
    )

    if not paths:
        raise RuntimeError(
            f"No image frames found in {case_dir}"
        )

    return paths


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    metadata_path = DATA_ROOT / "metadata.csv"

    metadata = pd.read_csv(
        metadata_path
    )

    case_col = "case_id"
    gt_col = "best_frame"

    case_dirs = sorted(
        [
            p
            for p in DATA_ROOT.iterdir()
            if (
                p.is_dir()
                and p.name.startswith("case_dev_")
            )
        ],
        key=lambda p: int(
            re.search(
                r"(\d+)$",
                p.name,
            ).group(1)
        ),
    )

    print("=" * 75)
    print("GENERATING DEVELOPMENT-SET VISUALIZATIONS")
    print("=" * 75)
    print()

    for case_dir in case_dirs:

        case_id = case_dir.name

        print(
            f"[{case_id}] Processing..."
        )

        # -------------------------------------------------------------
        # Load frames
        # -------------------------------------------------------------

        frame_paths = load_case_frames(
            case_dir
        )

        frames = [
            str(path)
            for path in frame_paths
        ]

        frame_ids = [
            frame_number(path)
            for path in frame_paths
        ]

        # -------------------------------------------------------------
        # Ground truth
        # -------------------------------------------------------------

        rows = metadata[
            metadata[case_col]
            .astype(str)
            .str.strip()
            == case_id
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one metadata row "
                f"for {case_id}, got {len(rows)}"
            )

        gt_frame = int(
            rows.iloc[0][gt_col]
        )

        if gt_frame not in frame_ids:
            raise ValueError(
                f"Ground-truth frame "
                f"{gt_frame} not found in {case_id}"
            )

        gt_index = frame_ids.index(
            gt_frame
        )

        # -------------------------------------------------------------
        # Run pipeline
        # -------------------------------------------------------------

        result = process_window(
            frames
        )

        frame_results = result[
            "frame_results"
        ]

        preprocessed_frames = result[
            "preprocessed_frames"
        ]

        vesselness_maps = result[
            "vesselness_maps"
        ]

        vessel_masks = result[
            "vessel_masks"
        ]

        best_index = result[
            "best_index"
        ]

        scores = np.asarray(
            [
                float(
                    frame_result[
                        "final_score"
                    ]
                )
                for frame_result
                in frame_results
            ]
        )

        predicted_frame = frame_ids[
            best_index
        ]

        # -------------------------------------------------------------
        # Output directory
        # -------------------------------------------------------------

        case_output = (
            OUTPUT_ROOT
            / case_id
        )

        case_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        labels = [
            f"frame_{frame_id:03d}"
            for frame_id in frame_ids
        ]

        # -------------------------------------------------------------
        # 1. Frame montage
        # -------------------------------------------------------------

        plot_score_grid(
            frames=preprocessed_frames,
            scores=scores,
            frame_labels=labels,
            gt_index=gt_index,
            selected_index=best_index,
            title=(
                f"{case_id} | Candidate frames and final scores"
            ),
            save_path=(
                case_output
                / "01_frame_scores.png"
            ),
        )

        # -------------------------------------------------------------
        # 2. Final score curve
        # -------------------------------------------------------------

        plot_score_curve(
            scores=scores,
            frame_ids=frame_ids,
            gt_index=gt_index,
            selected_index=best_index,
            title=(
                f"{case_id} | Final frame score"
            ),
            save_path=(
                case_output
                / "02_final_score_curve.png"
            ),
        )

        # -------------------------------------------------------------
        # 3. Branch scores
        # -------------------------------------------------------------

        plot_branch_scores(
            frame_results=frame_results,
            frame_ids=frame_ids,
            gt_index=gt_index,
            selected_index=best_index,
            title=(
                f"{case_id} | Vessel, quality and temporal branches"
            ),
            save_path=(
                case_output
                / "03_branch_scores.png"
            ),
        )

        # -------------------------------------------------------------
        # 4. Feature heatmap
        # -------------------------------------------------------------

        plot_feature_heatmap(
            frame_results=frame_results,
            frame_ids=frame_ids,
            title=(
                f"{case_id} | Normalized feature heatmap"
            ),
            save_path=(
                case_output
                / "04_feature_heatmap.png"
            ),
        )

        # -------------------------------------------------------------
        # 5. Temporal features
        # -------------------------------------------------------------

        plot_temporal_features(
            frame_results=frame_results,
            frame_ids=frame_ids,
            title=(
                f"{case_id} | Temporal feature behavior"
            ),
            save_path=(
                case_output
                / "05_temporal_features.png"
            ),
        )

        # -------------------------------------------------------------
        # 6. Weighted branch contributions
        # -------------------------------------------------------------

        plot_branch_contributions(
            frame_results=frame_results,
            frame_ids=frame_ids,
            vessel_weight=0.40,
            quality_weight=0.40,
            temporal_weight=0.20,
            gt_index=gt_index,
            selected_index=best_index,
            title=(
                f"{case_id} | Weighted score contributions"
            ),
            save_path=(
                case_output
                / "06_branch_contributions.png"
            ),
        )

        # -------------------------------------------------------------
        # 7. Selected-frame interpretability
        # -------------------------------------------------------------

        selected_score = float(
            frame_results[
                best_index
            ]["final_score"]
        )

        plot_interpretability(
            frame=preprocessed_frames[
                best_index
            ],
            vesselness=vesselness_maps[
                best_index
            ],
            vessel_mask=vessel_masks[
                best_index
            ],
            score=selected_score,
            title=(
                f"{case_id} | "
                f"Selected frame_{predicted_frame:03d}"
            ),
            save_path=(
                case_output
                / "07_selected_interpretability.png"
            ),
        )

        # -------------------------------------------------------------
        # 8. Top-K frames
        # -------------------------------------------------------------

        plot_top_k_frames(
            frames=preprocessed_frames,
            scores=scores,
            frame_ids=frame_ids,
            k=5,
            gt_frame=gt_frame,
            title=(
                f"{case_id} | Top-5 ranked frames"
            ),
            save_path=(
                case_output
                / "08_top5_frames.png"
            ),
        )

        # -------------------------------------------------------------
        # 9. Vessel + quality feature profiles
        # -------------------------------------------------------------

        plot_feature_profiles(
            frame_results=frame_results,
            frame_ids=frame_ids,
            gt_index=gt_index,
            selected_index=best_index,
            title=(
                f"{case_id} | Vessel and quality feature profiles"
            ),
            save_path=(
                case_output
                / "09_feature_profiles.png"
            ),
        )

        # -------------------------------------------------------------
        # 10. Complete dashboard
        # -------------------------------------------------------------

        plot_case_dashboard(
            frames=preprocessed_frames,
            frame_ids=frame_ids,
            frame_results=frame_results,
            vesselness=vesselness_maps,
            vessel_mask=vessel_masks,
            best_index=best_index,
            gt_index=gt_index,
            title=(
                f"{case_id} | Best-frame selection dashboard"
            ),
            save_path=(
                case_output
                / "10_case_dashboard.png"
            ),
        )

        # -------------------------------------------------------------
        # Console summary
        # -------------------------------------------------------------

        ranking = np.argsort(
            scores
        )[::-1]

        gt_rank = (
            int(
                np.where(
                    ranking == gt_index
                )[0][0]
            )
            + 1
        )

        print(
            f"    Frames   : {len(frames)}"
        )

        print(
            f"    GT       : frame_{gt_frame:03d}"
        )

        print(
            f"    Selected : frame_{predicted_frame:03d}"
        )

        print(
            f"    GT rank  : {gt_rank}"
        )

        print(
            f"    Output   : {case_output}"
        )

        print()

    print("=" * 75)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 75)
    print()
    print(
        f"Output directory:\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()