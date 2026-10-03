from pathlib import Path

from src.scene_renderer import render_scene


def render_all_scenes(
    scenes_dir: str | Path,
    output_dir: str | Path,
    fps: int = 30,
) -> None:
    scenes_path = Path(scenes_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for scene_id in range(1, 7):
        image_path = scenes_path / f"scene_{scene_id:02d}.png"

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing scene image: {image_path}")

        video_path = output_path / f"scene_{scene_id:02d}.mp4"

        render_scene(
            scene_id,
            image_path,
            video_path,
            fps=fps,
        )
