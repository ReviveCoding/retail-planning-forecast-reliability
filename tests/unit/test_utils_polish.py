"""Regression coverage for config path handling."""

from pathlib import Path

from src.utils import ROOT, load_yaml, project_path


def test_load_yaml_accepts_absolute_path(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("enabled: true\nseed: 42\n", encoding="utf-8")
    assert load_yaml(config) == {"enabled": True, "seed": 42}


def test_load_yaml_empty_document_uses_empty_mapping(tmp_path: Path) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    assert load_yaml(config) == {}


def test_project_path_preserves_repository_root() -> None:
    assert project_path() == ROOT
    assert project_path("configs", "pipeline.yaml") == ROOT / "configs/pipeline.yaml"
