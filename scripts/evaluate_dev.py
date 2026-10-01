from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(r"E:\OneDrive\Desktop\RNT Health Insights\coronary-best-frame")
DATA_ROOT = Path(
    r"E:\OneDrive\Desktop\Learn\Computer Vision\angiogram-keyframe\data\dev"
)

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------

sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import process_window


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_frame_number(path):
    """
    Extract numeric frame ID from names such as:
        frame_043.png
        frame_107.jpg
    """
    match = re.search(r"frame[_-]?(\d+)", path.stem.lower())

    if match:
        return int(match.group(1))

    return float("inf")


def load_metadata():
    """
    Automatically locate metadata.csv or metadata.xlsx/xls.
    """
    candidates = [
        DATA_ROOT / "metadata.csv",
        DATA_ROOT / "metadata.xlsx",
        DATA_ROOT / "metadata.xls",
    ]

    for path in candidates:
        if path.exists():
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path), path

            return pd.read_excel(path), path

    raise FileNotFoundError(
        f"Could not find metadata.csv/xlsx/xls in {DATA_ROOT}"
    )


def find_column(df, candidates):
    """
    Find a metadata column while allowing minor naming differences.
    """
    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


def extract_gt_frame(value):
    """
    Convert possible GT representations into an integer frame ID.

    Supported examples:
        47
        "47"
        "frame_047"
        "frame_047.png"
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(r"frame[_-]?(\d+)", text.lower())

    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)", text)

    if match:
        return int(match.group(1))

    return None


def rank_of_frame(scores, frame_ids, target_frame_id):
    """
    Rank is 1-based, with rank 1 being the highest score.
    """
    ranking = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    for rank, idx in enumerate(ranking, start=1):
        if frame_ids[idx] == target_frame_id:
            return rank

    return None


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main():

    metadata, metadata_path = load_metadata()

    print("=" * 70)
    print("CORONARY ANGIOGRAPHY BEST-FRAME DEVELOPMENT EVALUATION")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data root    : {DATA_ROOT}")
    print(f"Metadata     : {metadata_path}")
    print()

    print("Metadata columns:")
    print(list(metadata.columns))
    print()

    case_col = find_column(
        metadata,
        ["case_id", "case", "caseid"]
    )

    gt_col = find_column(
        metadata,
        [
            "best_frame",
            "best_frame_id",
            "gt_frame",
            "ground_truth",
            "ground_truth_frame",
        ],
    )

    if case_col is None:
        raise ValueError(
            "Could not identify the case ID column."
        )

    if gt_col is None:
        raise ValueError(
            "Could not identify the ground-truth best-frame column."
        )

    print(f"Case column       : {case_col}")
    print(f"Ground truth      : {gt_col}")
    print()

    case_results = []

    case_dirs = sorted(
        [
            p
            for p in DATA_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("case_dev_")
        ],
        key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)),
    )

    print(f"Development cases : {len(case_dirs)}")
    print()

    for case_dir in case_dirs:

        case_id = case_dir.name

        print("-" * 70)
        print(f"Processing {case_id}")

        # -------------------------------------------------------------
        # Load candidate frames
        # -------------------------------------------------------------

        frame_paths = sorted(
            [
                p
                for p in case_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=natural_frame_number,
        )

        if not frame_paths:
            raise RuntimeError(
                f"No image frames found in {case_dir}"
            )

        frames = [str(path) for path in frame_paths]

        frame_ids = [
            natural_frame_number(path)
            for path in frame_paths
        ]

        # -------------------------------------------------------------
        # Ground truth
        # -------------------------------------------------------------

        matching_rows = metadata[
            metadata[case_col].astype(str).str.strip() == case_id
        ]

        if len(matching_rows) != 1:
            raise ValueError(
                f"Expected exactly one metadata row for {case_id}, "
                f"found {len(matching_rows)}."
            )

        gt_frame = extract_gt_frame(
            matching_rows.iloc[0][gt_col]
        )

        if gt_frame is None:
            raise ValueError(
                f"Could not parse ground-truth frame for {case_id}."
            )

        if gt_frame not in frame_ids:
            raise ValueError(
                f"Ground-truth frame {gt_frame} is not present "
                f"in {case_id}."
            )

        # -------------------------------------------------------------
        # Run locked pipeline
        # -------------------------------------------------------------

        result = process_window(frames)

        best_index = result["best_index"]
        predicted_frame = frame_ids[best_index]

        scores = [
            float(frame_result["final_score"])
            for frame_result in result["frame_results"]
        ]

        gt_rank = rank_of_frame(
            scores,
            frame_ids,
            gt_frame,
        )

        # -------------------------------------------------------------
        # Metrics
        # -------------------------------------------------------------

        top1 = int(gt_rank <= 1)
        top3 = int(gt_rank <= 3)
        top5 = int(gt_rank <= 5)

        normalized_rank = (
            1.0
            - (gt_rank - 1) / max(len(frame_ids) - 1, 1)
        )

        print(f"Frames            : {len(frame_paths)}")
        print(f"Ground-truth      : frame_{gt_frame:03d}")
        print(f"Predicted         : frame_{predicted_frame:03d}")
        print(f"Ground-truth rank : {gt_rank}")
        print(f"Top-1             : {top1}")
        print(f"Top-3             : {top3}")
        print(f"Top-5             : {top5}")

        case_results.append(
            {
                "case_id": case_id,
                "num_frames": len(frame_paths),
                "ground_truth_frame": gt_frame,
                "predicted_frame": predicted_frame,
                "ground_truth_rank": gt_rank,
                "normalized_rank": normalized_rank,
                "top1": top1,
                "top3": top3,
                "top5": top5,
            }
        )

        # -------------------------------------------------------------
        # Save frame-level scores
        # -------------------------------------------------------------

        frame_score_df = pd.DataFrame(
            {
                "case_id": case_id,
                "frame_id": frame_ids,
                "score": scores,
                "is_ground_truth": [
                    frame_id == gt_frame
                    for frame_id in frame_ids
                ],
                "is_selected": [
                    frame_id == predicted_frame
                    for frame_id in frame_ids
                ],
            }
        )

        frame_score_df.to_csv(
            OUTPUT_DIR / f"{case_id}_scores.csv",
            index=False,
        )

    # -----------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(case_results)

    metrics = {
        "num_cases": len(results_df),
        "total_candidate_frames": int(
            results_df["num_frames"].sum()
        ),
        "mean_ground_truth_rank": (
            results_df["ground_truth_rank"].mean()
        ),
        "median_ground_truth_rank": (
            results_df["ground_truth_rank"].median()
        ),
        "mean_normalized_rank": (
            results_df["normalized_rank"].mean()
        ),
        "top1_accuracy": results_df["top1"].mean(),
        "top3_accuracy": results_df["top3"].mean(),
        "top5_accuracy": results_df["top5"].mean(),
    }

    metrics_df = pd.DataFrame(
        [metrics]
    )

    results_df.to_csv(
        OUTPUT_DIR / "dev_predictions.csv",
        index=False,
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "dev_metrics.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DEVELOPMENT-SET RESULTS")
    print("=" * 70)

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key:28s}: {value:.4f}")
        else:
            print(f"{key:28s}: {value}")

    print()
    print(f"Saved: {OUTPUT_DIR / 'dev_predictions.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'dev_metrics.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()