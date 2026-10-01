from pathlib import Path

import yaml
import numpy as np

from .preprocessing import load_grayscale, percentile_normalize
from .vesselness import compute_sato_vesselness
from .vessel_mask import create_vessel_mask
from .vessel_features import (
    largest_component_fraction,
    vessel_gradient_energy,
    local_vessel_background_contrast,
)
from .quality_features import (
    tenengrad,
    local_rms_contrast,
)
from .temporal_features import (
    compute_temporal_features,
)
from .normalization import normalize_feature_dicts
from .scoring import (
    compute_vessel_score,
    compute_quality_score,
    compute_temporal_score,
    compute_final_score,
    rank_indices,
)


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "default.yaml"
)


def load_config(config_path=None):
    """
    Load the pipeline configuration.

    Parameters
    ----------
    config_path : str | Path | None
        Configuration file path. If None, the submission's
        default.yaml is used.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

    return config


def validate_window(window):
    """
    Validate a candidate frame window.
    """
    if window is None:
        raise ValueError("window cannot be None.")

    if not isinstance(window, (list, tuple)):
        raise TypeError(
            "window must be a list or tuple of frames."
        )

    if len(window) == 0:
        raise ValueError(
            "window must contain at least one frame."
        )


def preprocess_image(image, config):
    """
    Apply the configured P1-P99 preprocessing.
    """
    gray = load_grayscale(image)

    normalization = config["preprocessing"]["normalization"]

    if normalization["method"] != "percentile":
        raise ValueError(
            "Final pipeline supports percentile normalization only."
        )

    return percentile_normalize(
        gray,
        lower_percentile=float(
            normalization["lower_percentile"]
        ),
        upper_percentile=float(
            normalization["upper_percentile"]
        ),
    )


def preprocess_window(window, config):
    """
    Preprocess every frame in the candidate window.
    """
    return [
        preprocess_image(frame, config)
        for frame in window
    ]


def extract_static_features(
    preprocessed_frames,
    config,
):
    """
    Extract the locked static features for every frame.
    """
    vessel_config = config["vesselness"]
    mask_config = config["vessel_mask"]

    vessel_feature_config = config["vessel_features"]
    quality_config = config["quality_features"]

    sigmas = tuple(
        float(sigma)
        for sigma in vessel_config["scales"]
    )

    black_ridges = bool(
        vessel_config["black_ridges"]
    )

    threshold_percentile = float(
        mask_config["threshold_percentile"]
    )

    min_component_size = int(
        mask_config["min_component_size"]
    )

    closing_radius = int(
        mask_config["morphology"]["radius"]
    )

    background_radius = int(
        vessel_feature_config[
            "local_background"
        ]["dilation_radius"]
    )

    rms_window = int(
        quality_config[
            "local_rms_contrast"
        ]["window_size"]
    )

    vessel_masks = []
    vesselness_maps = []
    frame_features = []

    for image in preprocessed_frames:

        vesselness = compute_sato_vesselness(
            image,
            sigmas=sigmas,
            black_ridges=black_ridges,
        )

        vessel_mask = create_vessel_mask(
            vesselness,
            threshold_percentile=threshold_percentile,
            min_component_size=min_component_size,
            closing_radius=closing_radius,
        )

        features = {
            "largest_component_fraction":
                largest_component_fraction(
                    vessel_mask
                ),

            "vessel_gradient_energy":
                vessel_gradient_energy(
                    image,
                    vessel_mask,
                ),

            "local_vessel_background_contrast":
                local_vessel_background_contrast(
                    image,
                    vessel_mask,
                    dilation_radius=background_radius,
                ),

            "tenengrad":
                tenengrad(image),

            "local_rms_contrast":
                local_rms_contrast(
                    image,
                    window_size=rms_window,
                ),
        }

        vessel_masks.append(vessel_mask)
        vesselness_maps.append(vesselness)
        frame_features.append(features)

    return (
        vessel_masks,
        vesselness_maps,
        frame_features,
    )


def extract_all_features(
    preprocessed_frames,
    config,
):
    """
    Extract static and temporal features for the full window.
    """
    (
        vessel_masks,
        vesselness_maps,
        frame_features,
    ) = extract_static_features(
        preprocessed_frames,
        config,
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

    temporal_features = compute_temporal_features(
        sharpness_values=sharpness_values,
        contrast_values=contrast_values,
        vessel_gradient_values=vessel_gradient_values,
        vessel_contrast_values=vessel_contrast_values,
    )

    complete_features = []

    for static, temporal in zip(
        frame_features,
        temporal_features,
    ):
        combined = dict(static)
        combined.update(temporal)
        complete_features.append(combined)

    return (
        vessel_masks,
        vesselness_maps,
        complete_features,
    )


def normalize_features(
    frame_features,
    config,
):
    """
    Apply configured case-wise P5-P95 normalization.
    """
    normalization = config["normalization"]

    feature_columns = []

    for feature in config["scoring"]["vessel_features"]:
        feature_columns.append(
            feature.replace("_norm", "")
        )

    for feature in config["scoring"]["quality_features"]:
        feature_columns.append(
            feature.replace("_norm", "")
        )

    for feature in config["scoring"]["temporal_features"]:
        feature_columns.append(
            feature.replace("_norm", "")
        )

    return normalize_feature_dicts(
        frame_feature_dicts=frame_features,
        feature_columns=feature_columns,
        lower_percentile=float(
            normalization["lower_percentile"]
        ),
        upper_percentile=float(
            normalization["upper_percentile"]
        ),
    )


def score_features(
    normalized_features,
    config,
):
    """
    Compute branch and final scores using configured weights.
    """
    weights = config["scoring"]["branch_weights"]

    vessel_weight = float(
        weights["vessel"]
    )

    quality_weight = float(
        weights["quality"]
    )

    temporal_weight = float(
        weights["temporal"]
    )

    total_weight = (
        vessel_weight
        + quality_weight
        + temporal_weight
    )

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "Branch weights must sum to 1.0."
        )

    results = []

    for features in normalized_features:

        vessel_score = compute_vessel_score(
            features
        )

        quality_score = compute_quality_score(
            features
        )

        temporal_score = compute_temporal_score(
            features
        )

        final_score = (
            vessel_weight * vessel_score
            + quality_weight * quality_score
            + temporal_weight * temporal_score
        )

        results.append(
            {
                "vessel_score": vessel_score,
                "quality_score": quality_score,
                "temporal_score": temporal_score,
                "final_score": float(final_score),
            }
        )

    return results


def process_window(
    window,
    config_path=None,
):
    """
    Run the complete locked classical-CV pipeline.

    Parameters
    ----------
    window : list | tuple
        Candidate angiographic frames.
    config_path : str | Path | None
        Optional configuration file.

    Returns
    -------
    dict
        Complete pipeline result.
    """
    validate_window(window)

    config = load_config(config_path)

    preprocessed_frames = preprocess_window(
        window,
        config,
    )

    (
        vessel_masks,
        vesselness_maps,
        frame_features,
    ) = extract_all_features(
        preprocessed_frames,
        config,
    )

    normalized_features = normalize_features(
        frame_features,
        config,
    )

    scores = score_features(
        normalized_features,
        config,
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
            "Final scores contain non-finite values."
        )

    best_index = int(
        np.argmax(final_scores)
    )

    ranking = list(
        np.argsort(
            -final_scores,
            kind="stable",
        )
    )

    frame_results = []

    for i, (features, score) in enumerate(
        zip(normalized_features, scores)
    ):
        frame_results.append(
            {
                "index": i,
                **features,
                **score,
            }
        )

    return {
        "best_index": best_index,
        "best_frame": window[best_index],
        "scores": scores,
        "ranking": ranking,
        "preprocessed_frames": preprocessed_frames,
        "vesselness_maps": vesselness_maps,
        "vessel_masks": vessel_masks,
        "frame_results": frame_results,
        "config": config,
    }


def select_best_frame(
    window,
    config_path=None,
):
    """
    Public submission API.

    Parameters
    ----------
    window : list | tuple
        Candidate angiographic frames.
    config_path : str | Path | None
        Optional configuration file.

    Returns
    -------
    best_frame
        Selected original input frame.

    scores : list[dict]
        Score dictionary for every frame.
    """
    result = process_window(
        window,
        config_path=config_path,
    )

    return (
        result["best_frame"],
        result["scores"],
    )
