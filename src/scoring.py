import numpy as np


VESSEL_FEATURES = (
    "largest_component_fraction_norm",
    "vessel_gradient_energy_norm",
    "local_vessel_background_contrast_norm",
)

QUALITY_FEATURES = (
    "tenengrad_norm",
    "local_rms_contrast_norm",
)

TEMPORAL_FEATURES = (
    "sharpness_ratio_prev_norm",
    "contrast_ratio_prev_norm",
    "sharpness_peak_ratio_norm",
    "vessel_gradient_peak_ratio_norm",
    "vessel_contrast_peak_ratio_norm",
)

VESSEL_WEIGHT = 0.40
QUALITY_WEIGHT = 0.40
TEMPORAL_WEIGHT = 0.20


def _mean_available(values):
    """
    Compute the mean of finite values.

    NaN values are ignored. This is required for the first
    frame of a temporal window, where previous-frame ratios
    are undefined.
    """
    values = np.asarray(values, dtype=np.float64)

    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return np.nan

    return float(np.mean(valid))


def compute_vessel_score(features):
    """
    Compute the normalized vessel branch score.

    S_V = mean(
        largest_component_fraction_norm,
        vessel_gradient_energy_norm,
        local_vessel_background_contrast_norm
    )
    """
    values = [
        features.get(name, np.nan)
        for name in VESSEL_FEATURES
    ]

    return _mean_available(values)


def compute_quality_score(features):
    """
    Compute the normalized image-quality branch score.

    S_Q = mean(
        tenengrad_norm,
        local_rms_contrast_norm
    )
    """
    values = [
        features.get(name, np.nan)
        for name in QUALITY_FEATURES
    ]

    return _mean_available(values)


def compute_temporal_score(features):
    """
    Compute the normalized temporal branch score.

    The first frame may contain NaN values for previous-frame
    ratios. Available temporal features are averaged rather
    than replacing undefined values with zero.
    """
    values = [
        features.get(name, np.nan)
        for name in TEMPORAL_FEATURES
    ]

    return _mean_available(values)


def compute_final_score(
    vessel_score,
    quality_score,
    temporal_score,
):
    """
    Compute the final weighted frame score.

    S_final = 0.40 * S_V
            + 0.40 * S_Q
            + 0.20 * S_T
    """
    scores = np.array(
        [
            vessel_score,
            quality_score,
            temporal_score,
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "All three branch scores must be finite before "
            "computing the final score."
        )

    return float(
        VESSEL_WEIGHT * vessel_score
        + QUALITY_WEIGHT * quality_score
        + TEMPORAL_WEIGHT * temporal_score
    )


def score_frame(features):
    """
    Compute all branch scores and the final score for one frame.

    Parameters
    ----------
    features : dict
        Dictionary containing normalized final features.

    Returns
    -------
    dict
        Vessel, quality, temporal, and final scores.
    """
    vessel_score = compute_vessel_score(features)
    quality_score = compute_quality_score(features)
    temporal_score = compute_temporal_score(features)

    final_score = compute_final_score(
        vessel_score=vessel_score,
        quality_score=quality_score,
        temporal_score=temporal_score,
    )

    return {
        "vessel_score": vessel_score,
        "quality_score": quality_score,
        "temporal_score": temporal_score,
        "final_score": final_score,
    }


def score_window(frame_features):
    """
    Score every frame in a candidate window.

    Parameters
    ----------
    frame_features : list[dict]
        Normalized feature dictionaries, one per frame.

    Returns
    -------
    list[dict]
        One score dictionary per frame.
    """
    if not frame_features:
        raise ValueError(
            "frame_features cannot be empty."
        )

    return [
        score_frame(features)
        for features in frame_features
    ]


def select_best_index(scores):
    """
    Select the index of the highest-scoring frame.

    Parameters
    ----------
    scores : list[dict]
        Frame score dictionaries containing "final_score".

    Returns
    -------
    int
        Zero-based index of the selected frame.
    """
    if not scores:
        raise ValueError(
            "scores cannot be empty."
        )

    final_scores = np.array(
        [
            score["final_score"]
            for score in scores
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(final_scores)):
        raise ValueError(
            "All final frame scores must be finite."
        )

    return int(np.argmax(final_scores))


def rank_indices(scores):
    """
    Rank frame indices from highest score to lowest score.

    Returns
    -------
    list[int]
        Zero-based frame indices.
    """
    if not scores:
        raise ValueError(
            "scores cannot be empty."
        )

    final_scores = np.array(
        [
            score["final_score"]
            for score in scores
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(final_scores)):
        raise ValueError(
            "All final frame scores must be finite."
        )

    return list(
        np.argsort(
            -final_scores,
            kind="stable",
        )
    )
