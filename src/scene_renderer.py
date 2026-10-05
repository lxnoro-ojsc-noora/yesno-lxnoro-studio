from pathlib import Path

from src.scene_motion import build_motion_matrices
from src.visuals import generate_master_to_30fps_video, load_image


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

    generate_master_to_30fps_video(
        image,
        start_matrix,
        end_matrix,
        output_path,
        master_fps=12,
        output_fps=fps,
        duration_seconds=10,
    )
