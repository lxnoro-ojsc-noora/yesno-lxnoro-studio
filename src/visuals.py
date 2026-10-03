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

def canny_edges(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low_threshold, high_threshold)

def affine_matrix(width: int, height: int, zoom: float = 1.0, pan_x: float = 0.0, pan_y: float = 0.0, rotation: float = 0.0) -> np.ndarray:
    center = (width / 2.0, height / 2.0)
    return cv2.getRotationMatrix2D(center, rotation, zoom).astype(np.float32) + np.array([[0.0, 0.0, pan_x], [0.0, 0.0, pan_y]], dtype=np.float32)

def apply_affine_motion(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def interpolate_motion(start: np.ndarray, end: np.ndarray, progress: float) -> np.ndarray:
    progress = float(np.clip(progress, 0.0, 1.0))
    return start + (end - start) * progress

def generate_motion_frame(image: np.ndarray, start_matrix: np.ndarray, end_matrix: np.ndarray, progress: float) -> np.ndarray:
    matrix = interpolate_motion(start_matrix, end_matrix, progress)
    return apply_affine_motion(image, matrix)

def generate_motion_sequence(image: np.ndarray, start_matrix: np.ndarray, end_matrix: np.ndarray, frame_count: int) -> list[np.ndarray]:
    if frame_count < 2:
        raise ValueError('frame_count must be at least 2')
    return [generate_motion_frame(image, start_matrix, end_matrix, i / (frame_count - 1)) for i in range(frame_count)]

def write_frame_sequence(frames: list[np.ndarray], output_path: str | Path, fps: int = 30) -> None:
    if not frames:
        raise ValueError('frames cannot be empty')
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise IOError(f'Could not open video writer: {output_path}')
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError('All frames must have identical dimensions')
            writer.write(frame)
    finally:
        writer.release()

