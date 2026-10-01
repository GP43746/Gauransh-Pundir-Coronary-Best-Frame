import numpy as np


def safe_ratio(current, reference, epsilon=1e-8):
    """
    Compute a numerically safe ratio:

        current / reference

    Parameters
    ----------
    current : float
        Current-frame measurement.
    reference : float
        Reference measurement.
    epsilon : float
        Small value used to avoid division by zero.

    Returns
    -------
    float
        Ratio.
    """
    if not np.isfinite(current) or not np.isfinite(reference):
        return np.nan

    if abs(reference) <= epsilon:
        return np.nan

    return float(current / reference)


def peak_ratio(value, peak_value, epsilon=1e-8):
    """
    Compute a value relative to the maximum value in the window.

        value / max(value)

    Parameters
    ----------
    value : float
        Current-frame feature value.
    peak_value : float
        Maximum feature value across the window.

    Returns
    -------
    float
        Peak ratio.
    """
    if not np.isfinite(value) or not np.isfinite(peak_value):
        return np.nan

    if abs(peak_value) <= epsilon:
        return np.nan

    return float(value / peak_value)


def compute_temporal_features(
    sharpness_values,
    contrast_values,
    vessel_gradient_values,
    vessel_contrast_values,
):
    """
    Compute the locked temporal feature set for one candidate window.

    The temporal features are:

        1. sharpness_ratio_prev
        2. contrast_ratio_prev
        3. sharpness_peak_ratio
        4. vessel_gradient_peak_ratio
        5. vessel_contrast_peak_ratio

    Parameters
    ----------
    sharpness_values : array-like
        Per-frame Tenengrad values.

    contrast_values : array-like
        Per-frame local RMS contrast values.

    vessel_gradient_values : array-like
        Per-frame vessel gradient energy values.

    vessel_contrast_values : array-like
        Per-frame local vessel-background contrast values.

    Returns
    -------
    list[dict]
        One temporal-feature dictionary per frame.
    """
    sharpness = np.asarray(
        sharpness_values,
        dtype=np.float64,
    )

    contrast = np.asarray(
        contrast_values,
        dtype=np.float64,
    )

    vessel_gradient = np.asarray(
        vessel_gradient_values,
        dtype=np.float64,
    )

    vessel_contrast = np.asarray(
        vessel_contrast_values,
        dtype=np.float64,
    )

    n = len(sharpness)

    if n == 0:
        raise ValueError("Temporal window cannot be empty.")

    if not (
        len(contrast) == n
        and len(vessel_gradient) == n
        and len(vessel_contrast) == n
    ):
        raise ValueError(
            "All temporal feature arrays must have the same length."
        )

    # Maximum values are computed over the complete candidate window.
    sharpness_peak = np.nanmax(sharpness)
    contrast_peak = np.nanmax(contrast)
    vessel_gradient_peak = np.nanmax(vessel_gradient)
    vessel_contrast_peak = np.nanmax(vessel_contrast)

    results = []

    for i in range(n):

        # Relative improvement from the immediately preceding frame.
        if i == 0:
            sharpness_ratio_prev = np.nan
            contrast_ratio_prev = np.nan
        else:
            sharpness_ratio_prev = safe_ratio(
                sharpness[i],
                sharpness[i - 1],
            )

            contrast_ratio_prev = safe_ratio(
                contrast[i],
                contrast[i - 1],
            )

        results.append(
            {
                "sharpness_ratio_prev": sharpness_ratio_prev,
                "contrast_ratio_prev": contrast_ratio_prev,
                "sharpness_peak_ratio": peak_ratio(
                    sharpness[i],
                    sharpness_peak,
                ),
                "vessel_gradient_peak_ratio": peak_ratio(
                    vessel_gradient[i],
                    vessel_gradient_peak,
                ),
                "vessel_contrast_peak_ratio": peak_ratio(
                    vessel_contrast[i],
                    vessel_contrast_peak,
                ),
            }
        )

    return results


def extract_temporal_features_from_frame_features(
    frame_features,
):
    """
    Compute temporal features from per-frame feature dictionaries.

    Each dictionary must contain:

        tenengrad
        local_rms_contrast
        vessel_gradient_energy
        local_vessel_background_contrast

    Parameters
    ----------
    frame_features : list[dict]
        Per-frame static feature dictionaries.

    Returns
    -------
    list[dict]
        Temporal features for each frame.
    """
    if not frame_features:
        raise ValueError("frame_features cannot be empty.")

    required = {
        "tenengrad",
        "local_rms_contrast",
        "vessel_gradient_energy",
        "local_vessel_background_contrast",
    }

    for i, features in enumerate(frame_features):
        missing = required - set(features.keys())

        if missing:
            raise KeyError(
                f"Frame {i} is missing required features: {sorted(missing)}"
            )

    sharpness_values = [
        features["tenengrad"]
        for features in frame_features
    ]

    contrast_values = [
        features["local_rms_contrast"]
        for features in frame_features
    ]

    vessel_gradient_values = [
        features["vessel_gradient_energy"]
        for features in frame_features
    ]

    vessel_contrast_values = [
        features["local_vessel_background_contrast"]
        for features in frame_features
    ]

    return compute_temporal_features(
        sharpness_values=sharpness_values,
        contrast_values=contrast_values,
        vessel_gradient_values=vessel_gradient_values,
        vessel_contrast_values=vessel_contrast_values,
    )
