"""Application settings. Everything is overridable with LOA_* environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOA_", env_file=".env", extra="ignore")

    data_dir: Path = Path.home() / "LocalOfficeAI"
    pack: str = "church"
    packs_dir: Path = REPO_ROOT / "packs"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"

    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"
    whisper_model: str = "small"

    pin: str = ""

    chunk_size: int = 800
    chunk_overlap: int = 120

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def pack_dir(self) -> Path:
        return self.packs_dir / self.pack

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.files_dir, self.outputs_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir = s.data_dir.expanduser()
    return s
