from pathlib import Path

import cv2

from src.scene_motion import build_motion_matrices
from src.visuals import generate_motion_video, load_image


def render_scene(
    scene_id: int,
    image_path: str | Path,
    output_path: str | Path,
    fps: int = 30,
) -> None:
    image = load_image(image_path)

    height, width = image.shape[:2]

    start_matrix, end_matrix = build_motion_matrices(
        scene_id,
        width,
        height,
    )

    frame_count = 10 * fps

    generate_motion_video(
        image,
        start_matrix,
        end_matrix,
        frame_count,
        output_path,
        fps=fps,
    )
