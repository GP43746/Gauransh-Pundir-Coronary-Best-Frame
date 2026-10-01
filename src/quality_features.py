import cv2
import numpy as np


def tenengrad(image):
    """
    Compute Tenengrad sharpness.

    Tenengrad is the mean squared Sobel gradient magnitude
    over the complete image.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image in [0, 1].

    Returns
    -------
    float
        Tenengrad sharpness measure.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}"
        )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient_squared = gx * gx + gy * gy

    return float(np.mean(gradient_squared))


def local_rms_contrast(
    image,
    window_size=15,
):
    """
    Compute mean local RMS contrast.

    For each pixel, local RMS contrast is estimated as the
    local standard deviation within a square neighborhood.

    The returned value is the spatial mean of the local
    standard deviation map.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image in [0, 1].
    window_size : int
        Size of the local neighborhood. Must be odd.

    Returns
    -------
    float
        Mean local RMS contrast.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}"
        )

    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(
            "window_size must be an odd integer >= 3."
        )

    mean = cv2.boxFilter(
        image,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    mean_squared = cv2.boxFilter(
        image * image,
        ddepth=-1,
        ksize=(window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    variance = np.maximum(
        mean_squared - mean * mean,
        0.0,
    )

    local_std = np.sqrt(variance)

    return float(np.mean(local_std))


def extract_quality_features(image):
    """
    Extract the complete locked image-quality feature set.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed normalized grayscale image.

    Returns
    -------
    dict
        Quality features.
    """
    return {
        "tenengrad": tenengrad(image),
        "local_rms_contrast": local_rms_contrast(
            image,
            window_size=15,
        ),
    }
