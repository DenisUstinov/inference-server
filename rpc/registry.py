import tomllib
from pathlib import Path

from pydantic import BaseModel


class ModelSpec(BaseModel):
    name: str
    enabled: bool
    method: str
    cli: str
    python: str
    timeout_seconds: int = 600


class Manifest(BaseModel):
    models: list[ModelSpec]


BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "models.manifest.toml"


def load_manifest() -> Manifest:
    with MANIFEST_PATH.open("rb") as manifest_file:
        return Manifest(**tomllib.load(manifest_file))


def find_model_by_method(method: str) -> ModelSpec:
    manifest = load_manifest()
    for model in manifest.models:
        if model.enabled and model.method == method:
            return model
    raise LookupError(f"no enabled model registered for method '{method}'")


def resolve_path(relative: str) -> Path:
    return BASE_DIR / relative
