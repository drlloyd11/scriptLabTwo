"""Configuration loading and sample configuration creation."""

from dataclasses import dataclass
from importlib import resources
import os
from pathlib import Path
from typing import Mapping, Tuple

import yaml


class ConfigError(ValueError):
    """Raised when a ScriptLab configuration cannot be read or validated."""


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    canonical_directories: Tuple[Path, ...]
    fanfiction_directory: Path
    storage_directory: Path
    embedding_model: str
    ollama_url: str


def _required_string(data: Mapping[str, object], key: str, config_path: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            "Configuration {!s} must define a non-empty string for {!r}.".format(
                config_path, key
            )
        )
    return value.strip()


def _required_strings(
    data: Mapping[str, object], key: str, config_path: Path
) -> Tuple[str, ...]:
    value = data.get(key)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []

    if not values:
        raise ConfigError(
            "Configuration {!s} must define {!r} as a non-empty path or list of "
            "paths.".format(config_path, key)
        )

    paths = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                "Configuration {!s} must define {!r}[{}] as a non-empty "
                "string.".format(config_path, key, index)
            )
        paths.append(item.strip())
    return tuple(paths)


def _resolve_path(value: str, config_directory: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_directory / path
    # Keep reparse points visible so source and storage paths can be checked
    # before any files are opened.
    return Path(os.path.abspath(os.fspath(path)))


def load_config(path: Path) -> ProjectConfig:
    """Load and validate a YAML configuration file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(
            "Configuration file not found: {!s}. Run `scriptlab init` to create "
            "a sample configuration.".format(config_path)
        )

    try:
        with config_path.open("r", encoding="utf-8-sig") as file:
            data = yaml.safe_load(file)
    except (OSError, UnicodeError) as exc:
        raise ConfigError(
            "Could not read configuration {!s}: {}".format(config_path, exc)
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(
            "Invalid YAML in configuration {!s}: {}".format(config_path, exc)
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            "Configuration {!s} must contain a YAML mapping.".format(config_path)
        )

    project_name = _required_string(data, "project_name", config_path)
    source_directories = data.get("source_directories")
    if not isinstance(source_directories, dict):
        raise ConfigError(
            "Configuration {!s} must define source_directories with canonical "
            "and fanfiction paths.".format(config_path)
        )

    canonical = _required_strings(source_directories, "canonical", config_path)
    fanfiction = _required_string(source_directories, "fanfiction", config_path)
    storage_directory = _required_string(data, "storage_directory", config_path)
    embedding_model = _required_string(data, "embedding_model", config_path)
    ollama_url = _required_string(data, "ollama_url", config_path)

    return ProjectConfig(
        project_name=project_name,
        canonical_directories=tuple(
            _resolve_path(directory, config_path.parent) for directory in canonical
        ),
        fanfiction_directory=_resolve_path(fanfiction, config_path.parent),
        storage_directory=_resolve_path(storage_directory, config_path.parent),
        embedding_model=embedding_model,
        ollama_url=ollama_url,
    )


def write_sample_config(path: Path) -> Path:
    """Write the packaged example config without overwriting an existing file."""
    config_path = Path(path).expanduser()
    try:
        sample = resources.read_text(
            "scriptlab", "config.example.yaml", encoding="utf-8"
        )
        with config_path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(sample)
    except FileExistsError as exc:
        raise ConfigError(
            "Refusing to overwrite existing configuration: {!s}".format(config_path)
        ) from exc
    except OSError as exc:
        raise ConfigError(
            "Could not create configuration {!s}: {}".format(config_path, exc)
        ) from exc
    return config_path
