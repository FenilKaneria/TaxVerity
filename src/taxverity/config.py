from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved from this file's location, which holds only while the package is
# installed in editable mode from the repo (src/taxverity/config.py). A
# non-editable install lands in site-packages and this points somewhere
# meaningless — such deployments must set the paths explicitly via env vars.
REPO_ROOT = Path(__file__).resolve().parents[2]

ENV_PREFIX = "TAXVERITY_"


class MissingSettingError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    corpus_pdf: Path = REPO_ROOT / "Income-tax-Act-2025.pdf"
    data_dir: Path = REPO_ROOT / "data"
    reports_dir: Path = REPO_ROOT / "reports"
    evals_dir: Path = REPO_ROOT / "evals"

    # Which index this process serves (Step 6.6). Both are named, never
    # inferred: an embedding set id is a serial, so it means nothing without the
    # corpus it was built for.
    serving_corpus_version: str | None = None
    serving_embedding_set_id: int | None = None

    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    jina_api_key: SecretStr | None = None
    database_url: SecretStr | None = None
    # Signs access tokens (Step 11.3). At least 32 bytes.
    jwt_secret: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def vectors_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "llm"

    def require(self, name: str) -> str:
        value = getattr(self, name)
        if value is None:
            raise MissingSettingError(
                f"{name} is not configured. Set {ENV_PREFIX}{name.upper()} in your "
                f"environment or in a .env file at the repository root. "
                f"See .env.example."
            )
        return value.get_secret_value() if isinstance(value, SecretStr) else value

    def resolve_corpus_pdf(self) -> Path:
        # The PDF is deliberately not tracked in git, so absence is an expected
        # first-run state rather than a corrupt installation.
        if not self.corpus_pdf.is_file():
            raise MissingSettingError(
                f"Corpus PDF not found at {self.corpus_pdf}. It is not tracked in "
                f"git; obtain it and place it at the repository root, or point "
                f"{ENV_PREFIX}CORPUS_PDF at it. See README.md."
            )
        return self.corpus_pdf
