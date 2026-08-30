from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rec709_to_linear(values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Decode Rec.709 encoded values while preserving finite extended range."""
    encoded = np.asarray(values, dtype=np.float32)
    return np.asarray(
        np.where(
            encoded < np.float32(0.081),
            encoded / np.float32(4.5),
            np.power(
                np.maximum(
                    (encoded + np.float32(0.099)) / np.float32(1.099),
                    np.float32(0.0),
                ),
                np.float32(1.0 / 0.45),
            ),
        ),
        dtype=np.float32,
    )


def linear_to_rec709(values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Encode linear-light values to Rec.709 without clipping extended range."""
    linear = np.asarray(values, dtype=np.float32)
    return np.asarray(
        np.where(
            linear < np.float32(0.018),
            linear * np.float32(4.5),
            np.float32(1.099)
            * np.power(np.maximum(linear, np.float32(0.0)), np.float32(0.45))
            - np.float32(0.099),
        ),
        dtype=np.float32,
    )
