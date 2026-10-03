from pathlib import Path

from moviepy import VideoFileClip, concatenate_videoclips


def compose_trailer(
    scenes_dir: str | Path,
    output_path: str | Path,
) -> None:
    scenes_path = Path(scenes_dir)
    output_file = Path(output_path)

    clips = []

    try:
        for scene_id in range(1, 7):
            scene_path = scenes_path / f"scene_{scene_id:02d}.mp4"

            if not scene_path.is_file():
                raise FileNotFoundError(f"Missing scene video: {scene_path}")

            clips.append(VideoFileClip(str(scene_path)))

        final_clip = concatenate_videoclips(
            clips,
            method="compose",
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)

        final_clip.write_videofile(
            str(output_file),
            codec="libx264",
            audio=False,
        )

        final_clip.close()

    finally:
        for clip in clips:
            clip.close()
