from pathlib import Path

import cv2
import numpy as np
from color_matcher import ColorMatcher


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return image


def color_match(
    source: np.ndarray,
    reference: np.ndarray,
    method: str = "mkl",
) -> np.ndarray:
    result = ColorMatcher().transfer(
        src=source,
        ref=reference,
        method=method,
    )

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def save_image(image: np.ndarray, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Could not write image: {output_path}")
