import cv2
import numpy as np


def largest_component_fraction(mask):
    """
    Fraction of all vessel-mask pixels belonging to the largest
    connected component.

    Higher values indicate a more dominant connected vessel structure.

    Returns
    -------
    float
    """
    mask = (np.asarray(mask) > 0).astype(np.uint8)

    total_area = int(mask.sum())

    if total_area == 0:
        return 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if num_labels <= 1:
        return 0.0

    component_areas = stats[1:, cv2.CC_STAT_AREA]

    largest_area = float(np.max(component_areas))

    return largest_area / float(total_area)


def vessel_gradient_energy(image, mask):
    """
    Mean squared Sobel gradient magnitude inside the vessel mask.

    This measures the strength of local intensity transitions
    associated with the candidate vessel region.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image in [0, 1].
    mask : np.ndarray
        Binary vessel mask.

    Returns
    -------
    float
    """
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask) > 0

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, got shape {image.shape}"
        )

    if mask.shape != image.shape:
        raise ValueError(
            "Image and mask must have identical spatial dimensions."
        )

    if not np.any(mask):
        return 0.0

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

    return float(np.mean(gradient_squared[mask]))


def local_vessel_background_contrast(
    image,
    mask,
    dilation_radius=3,
):
    """
    Estimate local vessel-to-background contrast.

    The background region is obtained by dilating the vessel mask
    and taking the surrounding pixels that are not part of the
    vessel mask.

    Contrast is measured as:

        |mean(vessel) - mean(local_background)|

    Parameters
    ----------
    image : np.ndarray
        Preprocessed grayscale image in [0, 1].
    mask : np.ndarray
        Binary vessel mask.
    dilation_radius : int
        Radius used to define the local surrounding region.

    Returns
    -------
    float
    """
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask) > 0

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, got shape {image.shape}"
        )

    if mask.shape != image.shape:
        raise ValueError(
            "Image and mask must have identical spatial dimensions."
        )

    if not np.any(mask):
        return 0.0

    if dilation_radius < 1:
        raise ValueError("dilation_radius must be >= 1.")

    vessel_mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * dilation_radius + 1,
            2 * dilation_radius + 1,
        ),
    )

    dilated_mask = cv2.dilate(
        vessel_mask,
        kernel,
    ).astype(bool)

    background_mask = dilated_mask & (~mask)

    if not np.any(background_mask):
        return 0.0

    vessel_mean = float(np.mean(image[mask]))
    background_mean = float(np.mean(image[background_mask]))

    return abs(vessel_mean - background_mean)


def extract_vessel_features(image, mask):
    """
    Extract the complete locked vessel feature set.

    Parameters
    ----------
    image : np.ndarray
        Preprocessed normalized grayscale image.
    mask : np.ndarray
        Final binary vessel mask.

    Returns
    -------
    dict
        Vessel features.
    """
    return {
        "largest_component_fraction": (
            largest_component_fraction(mask)
        ),
        "vessel_gradient_energy": (
            vessel_gradient_energy(image, mask)
        ),
        "local_vessel_background_contrast": (
            local_vessel_background_contrast(image, mask)
        ),
    }
