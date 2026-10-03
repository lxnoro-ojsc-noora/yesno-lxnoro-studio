from dataclasses import dataclass

import numpy as np

from src.visuals import affine_matrix


@dataclass(frozen=True)
class SceneMotion:
    zoom_start: float
    zoom_end: float
    pan_x_start: float
    pan_x_end: float
    pan_y_start: float
    pan_y_end: float
    rotation_start: float
    rotation_end: float


SCENE_MOTIONS: dict[int, SceneMotion] = {
    1: SceneMotion(1.00, 1.05, 0.0, 8.0, 0.0, -4.0, 0.0, 0.5),
    2: SceneMotion(1.02, 1.08, -8.0, 8.0, 0.0, 0.0, -0.5, 0.5),
    3: SceneMotion(1.06, 1.12, -20.0, 20.0, 4.0, -6.0, 0.5, -0.5),
    4: SceneMotion(1.03, 1.10, 0.0, 0.0, 8.0, -8.0, 0.0, 0.0),
    5: SceneMotion(1.08, 1.14, -12.0, 12.0, -4.0, 4.0, -0.5, 0.5),
    6: SceneMotion(1.00, 1.06, 0.0, 0.0, 0.0, -10.0, 0.0, 0.0),
}


def build_motion_matrices(
    scene_id: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scene_id not in SCENE_MOTIONS:
        raise ValueError(f"Unknown scene_id: {scene_id}")

    motion = SCENE_MOTIONS[scene_id]

    start_matrix = affine_matrix(
        width,
        height,
        motion.zoom_start,
        motion.pan_x_start,
        motion.pan_y_start,
        motion.rotation_start,
    )

    end_matrix = affine_matrix(
        width,
        height,
        motion.zoom_end,
        motion.pan_x_end,
        motion.pan_y_end,
        motion.rotation_end,
    )

    return start_matrix, end_matrix
