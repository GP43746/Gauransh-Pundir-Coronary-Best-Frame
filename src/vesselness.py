import numpy as np
from skimage.filters import sato


SATO_SCALES = (1.0, 2.0, 3.0)


def compute_sato_vesselness(
    image,
    sigmas=SATO_SCALES,
    black_ridges=True,
):
    """
    Compute Sato vesselness for a normalized grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image, preferably normalized to [0, 1].
    sigmas : tuple[float, ...]
        Gaussian scales used by the Sato filter.
        Final pipeline uses (1, 2, 3).
    black_ridges : bool
        Detect dark tubular structures on a bright background.

    Returns
    -------
    np.ndarray
        Sato vesselness response as float32.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, got shape {image.shape}"
        )

    if len(sigmas) == 0:
        raise ValueError("sigmas must contain at least one scale.")

    if any(float(sigma) <= 0 for sigma in sigmas):
        raise ValueError("All Sato scales must be positive.")

    vesselness = sato(
        image,
        sigmas=tuple(float(sigma) for sigma in sigmas),
        black_ridges=black_ridges,
    )

    return np.asarray(vesselness, dtype=np.float32)


def compute_final_vesselness(image):
    """
    Compute vesselness using the locked final configuration.

    Final configuration:
        Sato
        scales = [1, 2, 3]
        black_ridges = True
    """
    return compute_sato_vesselness(
        image,
        sigmas=SATO_SCALES,
        black_ridges=True,
    )
