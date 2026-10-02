import json
from pathlib import Path

from src.models import Trailer


def load_screenplay(path: str | Path) -> Trailer:
    screenplay_path = Path(path)

    with screenplay_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    return Trailer.model_validate(data)
