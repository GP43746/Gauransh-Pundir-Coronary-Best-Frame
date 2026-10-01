from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(
    r"E:\OneDrive\Desktop\RNT Health Insights\coronary-best-frame"
)

DATA_ROOT = Path(
    r"E:\OneDrive\Desktop\Learn\Computer Vision\angiogram-keyframe\data\dev"
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dev_visualizations"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import process_window
from src.visualization import plot_score_grid, plot_score_curve


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def frame_number(path):
    match = re.search(r"frame[_-]?(\d+)", path.stem.lower())

    if match:
        return int(match.group(1))

    raise ValueError(f"Could not extract frame number from {path}")


def main():

    metadata_path = DATA_ROOT / "metadata.csv"
    metadata = pd.read_csv(metadata_path)

    # Expected metadata columns from the assignment dataset.
    case_col = "case_id"
    gt_col = "best_frame"

    case_dirs = sorted(
        [
            p for p in DATA_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("case_dev_")
        ],
        key=lambda p: int(
            re.search(r"(\d+)$", p.name).group(1)
        ),
    )

    for case_dir in case_dirs:

        case_id = case_dir.name

        print(f"Visualizing {case_id}...")

        frame_paths = sorted(
            [
                p for p in case_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=frame_number,
        )

        frames = [str(p) for p in frame_paths]
        frame_ids = [frame_number(p) for p in frame_paths]

        row = metadata[
            metadata[case_col].astype(str).str.strip() == case_id
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one metadata row for {case_id}"
            )

        gt_frame = int(row.iloc[0][gt_col])

        result = process_window(frames)

        scores = result["frame_results"]

        best_index = result["best_index"]
        predicted_frame = frame_ids[best_index]

        case_output = OUTPUT_ROOT / case_id
        case_output.mkdir(parents=True, exist_ok=True)

        labels = [
            f"frame_{frame_id:03d}"
            for frame_id in frame_ids
        ]

        # ---------------------------------------------------------
        # Frame grid with scores
        # ---------------------------------------------------------

        plot_score_grid(
            frames=result["preprocessed_frames"],
            scores=scores,
            frame_labels=labels,
            gt_index=frame_ids.index(gt_frame),
            title=(
                f"{case_id} | "
                f"Selected: frame_{predicted_frame:03d} | "
                f"GT: frame_{gt_frame:03d}"
            ),
            save_path=case_output / "frame_scores.png",
        )

        # ---------------------------------------------------------
        # Score curve
        # ---------------------------------------------------------

        plot_score_curve(
            scores=scores,
            gt_index=frame_ids.index(gt_frame),
            title=(
                f"{case_id} | Final frame score"
            ),
            save_path=case_output / "score_curve.png",
        )

        print(
            f"  GT       : frame_{gt_frame:03d}"
        )
        print(
            f"  Selected : frame_{predicted_frame:03d}"
        )

    print()
    print(
        f"Visualizations saved to:\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()