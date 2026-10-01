from pathlib import Path

import cv2

from src.pipeline import select_best_frame


def load_frames_from_directory(directory):
    """
    Load candidate frames from a directory in filename order.
    """
    directory = Path(directory)

    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    if not paths:
        raise ValueError(
            f"No image frames found in: {directory}"
        )

    frames = []

    for path in paths:
        frame = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if frame is None:
            raise ValueError(
                f"Could not read frame: {path}"
            )

        frames.append(frame)

    return paths, frames


def main():
    # Replace this with a directory containing one candidate window.
    window_directory = Path(
        "examples/test_window"
    )

    paths, window = load_frames_from_directory(
        window_directory
    )

    best_frame, scores = select_best_frame(
        window
    )

    best_index = max(
        range(len(scores)),
        key=lambda i: scores[i]["final_score"],
    )

    print("Number of candidate frames:", len(window))
    print(
        "Selected frame:",
        paths[best_index].name,
    )

    print("\nScores:")

    for i, (path, score) in enumerate(
        zip(paths, scores)
    ):
        print(
            f"{path.name}: "
            f"final={score['final_score']:.4f}, "
            f"vessel={score['vessel_score']:.4f}, "
            f"quality={score['quality_score']:.4f}, "
            f"temporal={score['temporal_score']:.4f}"
        )


if __name__ == "__main__":
    main()
