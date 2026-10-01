import cv2
import numpy as np


def load_grayscale(image):
    """
    Load an angiographic frame as a grayscale float32 image.

    Parameters
    ----------
    image : str | np.ndarray
        Image file path or already-loaded image.

    Returns
    -------
    np.ndarray
        Grayscale float32 image.
    """
    if isinstance(image, (str, bytes)):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not load image: {image}")

    elif isinstance(image, np.ndarray):
        img = image.copy()

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif img.ndim != 2:
            raise ValueError(
                f"Expected 2D grayscale or 3D color image, got shape {img.shape}"
            )

    else:
        raise TypeError(
            "image must be a file path or numpy.ndarray"
        )

    return img.astype(np.float32)


def percentile_normalize(
    image,
    lower_percentile=1.0,
    upper_percentile=99.0,
):
    """
    Robust percentile normalization.

    The image is normalized using the specified lower and upper
    intensity percentiles and clipped to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    lower_percentile : float
        Lower percentile. Default: 1.
    upper_percentile : float
        Upper percentile. Default: 99.

    Returns
    -------
    np.ndarray
        Normalized float32 image in [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}"
        )

    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower_percentile < upper_percentile <= 100"
        )

    p_low = np.percentile(image, lower_percentile)
    p_high = np.percentile(image, upper_percentile)

    # Degenerate image: avoid division by zero.
    if p_high <= p_low:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - p_low) / (p_high - p_low)

    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def preprocess_frame(image):
    """
    Apply the final locked preprocessing pipeline.

    Pipeline:
        Load grayscale
        -> P1-P99 percentile normalization

    Parameters
    ----------
    image : str | np.ndarray
        Input angiographic frame.

    Returns
    -------
    np.ndarray
        Preprocessed image in [0, 1].
    """
    gray = load_grayscale(image)

    normalized = percentile_normalize(
        gray,
        lower_percentile=1.0,
        upper_percentile=99.0,
    )

    return normalized
