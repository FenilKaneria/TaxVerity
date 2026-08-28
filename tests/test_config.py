from pathlib import Path

import pytest

from taxverity.config import REPO_ROOT, MissingSettingError, Settings


# A real .env at the repo root would otherwise leak into these assertions.
def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_path_defaults_resolve_under_the_repo_root():
    settings = make_settings()
    assert settings.corpus_pdf == REPO_ROOT / "Income-tax-Act-2025.pdf"
    assert settings.data_dir == REPO_ROOT / "data"
    assert settings.reports_dir == REPO_ROOT / "reports"
    assert settings.evals_dir == REPO_ROOT / "evals"
    assert settings.interim_dir == REPO_ROOT / "data" / "interim"


def test_secrets_default_to_unset():
    settings = make_settings()
    assert settings.groq_api_key is None
    assert settings.database_url is None


def test_env_var_overrides_a_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TAXVERITY_REPORTS_DIR", str(tmp_path / "elsewhere"))
    assert make_settings().reports_dir == tmp_path / "elsewhere"


def test_env_var_overrides_a_secret(monkeypatch):
    monkeypatch.setenv("TAXVERITY_GROQ_API_KEY", "sk-test-value")
    assert make_settings().require("groq_api_key") == "sk-test-value"


def test_interim_dir_follows_a_data_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TAXVERITY_DATA_DIR", str(tmp_path))
    assert make_settings().interim_dir == tmp_path / "interim"


def test_env_file_is_read(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TAXVERITY_LANGFUSE_HOST=http://example.invalid\n")
    monkeypatch.chdir(tmp_path)
    assert Settings(_env_file=env_file).langfuse_host == "http://example.invalid"


def test_env_var_wins_over_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TAXVERITY_LANGFUSE_HOST=http://from-file.invalid\n")
    monkeypatch.setenv("TAXVERITY_LANGFUSE_HOST", "http://from-env.invalid")
    assert Settings(_env_file=env_file).langfuse_host == "http://from-env.invalid"


def test_require_raises_clearly_when_a_secret_is_missing():
    with pytest.raises(MissingSettingError) as excinfo:
        make_settings().require("groq_api_key")
    message = str(excinfo.value)
    assert "TAXVERITY_GROQ_API_KEY" in message
    assert ".env.example" in message


def test_secret_values_are_not_exposed_by_repr(monkeypatch):
    monkeypatch.setenv("TAXVERITY_GROQ_API_KEY", "sk-should-not-leak")
    settings = make_settings()
    assert "sk-should-not-leak" not in repr(settings)
    assert "sk-should-not-leak" not in str(settings.groq_api_key)


def test_resolve_corpus_pdf_returns_the_path_when_present(tmp_path):
    pdf = tmp_path / "Act.pdf"
    pdf.write_bytes(b"%PDF-1.7\n")
    assert make_settings(corpus_pdf=pdf).resolve_corpus_pdf() == pdf


def test_resolve_corpus_pdf_raises_clearly_when_absent(tmp_path):
    missing = tmp_path / "absent.pdf"
    with pytest.raises(MissingSettingError) as excinfo:
        make_settings(corpus_pdf=missing).resolve_corpus_pdf()
    message = str(excinfo.value)
    assert str(missing) in message
    assert "README.md" in message


def test_a_directory_is_not_accepted_as_the_corpus_pdf(tmp_path):
    with pytest.raises(MissingSettingError):
        make_settings(corpus_pdf=tmp_path).resolve_corpus_pdf()


def test_unknown_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("TAXVERITY_NOT_A_REAL_SETTING", "x")
    assert isinstance(make_settings().data_dir, Path)
