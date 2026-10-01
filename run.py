from pathlib import Path
import argparse

import cv2

from src.pipeline import select_best_frame


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def load_window_from_directory(directory):
    """
    Load all image frames from a candidate-window directory
    in filename order.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {directory}"
        )

    image_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(
            f"No supported image files found in {directory}"
        )

    frames = []

    for path in image_paths:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {path}"
            )

        frames.append(image)

    return image_paths, frames


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classical CV best-frame selection "
            "from coronary angiography."
        )
    )

    parser.add_argument(
        "window_dir",
        type=str,
        help="Directory containing one candidate frame window.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to a YAML configuration file. "
            "Defaults to configs/default.yaml."
        ),
    )

    args = parser.parse_args()

    image_paths, frames = load_window_from_directory(
        args.window_dir
    )

    best_frame, scores = select_best_frame(
        frames,
        config_path=args.config,
    )

    best_index = max(
        range(len(scores)),
        key=lambda i: scores[i]["final_score"],
    )

    print()
    print("=" * 90)
    print("CORONARY ANGIOGRAPHY BEST-FRAME SELECTION")
    print("=" * 90)

    print(f"Number of frames : {len(frames)}")
    print(f"Selected index   : {best_index}")
    print(f"Selected frame   : {image_paths[best_index].name}")
    print()

    print(
        f"{'Index':>5}  "
        f"{'Frame':<30}  "
        f"{'Vessel':>8}  "
        f"{'Quality':>8}  "
        f"{'Temporal':>8}  "
        f"{'Final':>8}"
    )

    print("-" * 90)

    for i, (path, score) in enumerate(
        zip(image_paths, scores)
    ):
        marker = " <-- SELECTED" if i == best_index else ""

        print(
            f"{i:>5}  "
            f"{path.name:<30}  "
            f"{score['vessel_score']:>8.4f}  "
            f"{score['quality_score']:>8.4f}  "
            f"{score['temporal_score']:>8.4f}  "
            f"{score['final_score']:>8.4f}"
            f"{marker}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
