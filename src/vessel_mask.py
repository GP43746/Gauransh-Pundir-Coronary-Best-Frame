import cv2
import numpy as np
from skimage.morphology import disk


VESSEL_THRESHOLD_PERCENTILE = 90.0
MIN_COMPONENT_SIZE = 50
CLOSING_RADIUS = 1


def percentile_threshold(vesselness, percentile=VESSEL_THRESHOLD_PERCENTILE):
    """
    Threshold a vesselness image using a percentile of its response.

    Parameters
    ----------
    vesselness : np.ndarray
        2D vesselness response.
    percentile : float
        Threshold percentile. Final pipeline uses P90.

    Returns
    -------
    np.ndarray
        Binary uint8 mask.
    """
    vesselness = np.asarray(vesselness, dtype=np.float32)

    if vesselness.ndim != 2:
        raise ValueError(
            f"Expected a 2D vesselness image, got shape {vesselness.shape}"
        )

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100.")

    threshold = np.percentile(vesselness, percentile)

    mask = vesselness >= threshold

    return mask.astype(np.uint8)


def remove_small_components(mask, min_size=MIN_COMPONENT_SIZE):
    """
    Remove connected components smaller than min_size pixels.

    Uses 8-connectivity so diagonally connected vessel structures
    remain connected.
    """
    mask = np.asarray(mask, dtype=np.uint8)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got shape {mask.shape}"
        )

    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary)

    # Label 0 is the background.
    for label in range(1, num_labels):
        component_area = stats[label, cv2.CC_STAT_AREA]

        if component_area >= min_size:
            cleaned[labels == label] = 1

    return cleaned


def morphological_close(mask, radius=CLOSING_RADIUS):
    """
    Apply morphological closing using a disk-shaped structuring element.

    Closing helps connect small gaps in candidate vessel structures.
    """
    mask = np.asarray(mask, dtype=np.uint8)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got shape {mask.shape}"
        )

    if radius < 1:
        raise ValueError("radius must be >= 1.")

    # skimage.disk returns a boolean structuring element.
    footprint = disk(radius).astype(np.uint8)

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        footprint,
    )

    return (closed > 0).astype(np.uint8)


def create_vessel_mask(
    vesselness,
    threshold_percentile=VESSEL_THRESHOLD_PERCENTILE,
    min_component_size=MIN_COMPONENT_SIZE,
    closing_radius=CLOSING_RADIUS,
):
    """
    Create the final locked vessel mask.

    Pipeline:
        P90 threshold
        -> remove components < 50 px
        -> morphological closing with disk radius 1

    Parameters
    ----------
    vesselness : np.ndarray
        Sato vesselness response.
    threshold_percentile : float
        Vesselness threshold percentile. Default: 90.
    min_component_size : int
        Minimum retained connected-component size. Default: 50.
    closing_radius : int
        Morphological closing radius. Default: 1.

    Returns
    -------
    np.ndarray
        Binary uint8 vessel mask.
    """
    mask = percentile_threshold(
        vesselness,
        percentile=threshold_percentile,
    )

    mask = remove_small_components(
        mask,
        min_size=min_component_size,
    )

    mask = morphological_close(
        mask,
        radius=closing_radius,
    )

    return mask


def create_final_vessel_mask(vesselness):
    """
    Create the vessel mask using the locked final configuration.
    """
    return create_vessel_mask(
        vesselness,
        threshold_percentile=90.0,
        min_component_size=50,
        closing_radius=1,
    )
