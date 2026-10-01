import numpy as np
import pandas as pd


def robust_percentile_normalize(
    values,
    lower_percentile=5.0,
    upper_percentile=95.0,
):
    """
    Robust percentile-based min-max normalization.

    Formula:

        x_norm = clip(
            (x - P5) / (P95 - P5),
            0,
            1
        )

    Parameters
    ----------
    values : array-like
        Feature values.
    lower_percentile : float
        Lower normalization percentile. Default: 5.
    upper_percentile : float
        Upper normalization percentile. Default: 95.

    Returns
    -------
    np.ndarray
        Normalized values in [0, 1].
    """
    values = np.asarray(values, dtype=np.float64)

    if not (
        0 <= lower_percentile < upper_percentile <= 100
    ):
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower_percentile < upper_percentile <= 100."
        )

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return np.full(
            values.shape,
            np.nan,
            dtype=np.float64,
        )

    p_low = np.percentile(
        finite_values,
        lower_percentile,
    )

    p_high = np.percentile(
        finite_values,
        upper_percentile,
    )

    # Constant feature within the window.
    if p_high <= p_low:
        normalized = np.zeros_like(
            values,
            dtype=np.float64,
        )

        normalized[np.isfinite(values)] = 0.5

        return normalized

    normalized = (
        values - p_low
    ) / (
        p_high - p_low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    normalized[~np.isfinite(values)] = np.nan

    return normalized


def normalize_dataframe_features(
    dataframe,
    feature_columns,
    case_column="case_id",
    lower_percentile=5.0,
    upper_percentile=95.0,
):
    """
    Apply case-wise P5-P95 normalization to selected features.

    Each candidate window/case is normalized independently.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Data containing one row per candidate frame.
    feature_columns : list[str]
        Features to normalize.
    case_column : str
        Column identifying the candidate window/case.
    lower_percentile : float
        Lower normalization percentile. Default: 5.
    upper_percentile : float
        Upper normalization percentile. Default: 95.

    Returns
    -------
    pandas.DataFrame
        Copy of dataframe with normalized columns appended using
        the suffix "_norm".
    """
    if case_column not in dataframe.columns:
        raise KeyError(
            f"Missing case column: {case_column}"
        )

    missing = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise KeyError(
            f"Missing feature columns: {missing}"
        )

    result = dataframe.copy()

    for feature in feature_columns:

        normalized_column = f"{feature}_norm"

        result[normalized_column] = (
            result
            .groupby(case_column, group_keys=False)[feature]
            .transform(
                lambda values: robust_percentile_normalize(
                    values.to_numpy(),
                    lower_percentile=lower_percentile,
                    upper_percentile=upper_percentile,
                )
            )
        )

    return result


def normalize_feature_dicts(
    frame_feature_dicts,
    feature_columns,
    lower_percentile=5.0,
    upper_percentile=95.0,
):
    """
    Normalize features stored as a list of dictionaries.

    This helper is useful for the final inference pipeline where
    all frames from one candidate window are already represented
    as dictionaries.

    Parameters
    ----------
    frame_feature_dicts : list[dict]
        One feature dictionary per frame.
    feature_columns : list[str]
        Features to normalize.

    Returns
    -------
    list[dict]
        Copies of the dictionaries with "_norm" features added.
    """
    if not frame_feature_dicts:
        raise ValueError(
            "frame_feature_dicts cannot be empty."
        )

    result = [
        dict(features)
        for features in frame_feature_dicts
    ]

    for feature in feature_columns:

        values = np.array(
            [
                features.get(feature, np.nan)
                for features in result
            ],
            dtype=np.float64,
        )

        normalized = robust_percentile_normalize(
            values,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

        normalized_column = f"{feature}_norm"

        for features, value in zip(
            result,
            normalized,
        ):
            features[normalized_column] = float(value)

    return result
