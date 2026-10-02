from pydantic import BaseModel, Field, model_validator


class Scene(BaseModel):
    scene_id: int = Field(ge=1, le=6)
    duration_seconds: int = Field(default=10, ge=1)
    title: str
    description: str
    dialogue: str = ""
    visual_prompt: str
    mood: str

    @model_validator(mode="after")
    def validate_duration(self):
        if self.duration_seconds != 10:
            raise ValueError("Every scene must be exactly 10 seconds")
        return self


class Trailer(BaseModel):
    title: str
    total_duration_seconds: int = Field(default=60, ge=1)
    scenes: list[Scene]

    @model_validator(mode="after")
    def validate_structure(self):
        if self.total_duration_seconds != 60:
            raise ValueError("Trailer must be exactly 60 seconds")

        if len(self.scenes) != 6:
            raise ValueError("Trailer must contain exactly 6 scenes")

        scene_ids = [scene.scene_id for scene in self.scenes]
        if scene_ids != [1, 2, 3, 4, 5, 6]:
            raise ValueError("Scene IDs must be exactly 1 through 6")

        if sum(scene.duration_seconds for scene in self.scenes) != 60:
            raise ValueError("Scene durations must total exactly 60 seconds")

        return self
