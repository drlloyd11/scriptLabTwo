import pytest

from scriptlab.config import ConfigError, load_config


def test_missing_config_has_actionable_error(tmp_path):
    with pytest.raises(ConfigError, match="Run `scriptlab init`"):
        load_config(tmp_path / "missing.yaml")


def test_malformed_yaml_has_readable_error(tmp_path):
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text("project_name: [broken", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(config_path)


def test_non_mapping_config_is_rejected(tmp_path):
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text("- item\n- item", encoding="utf-8")

    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(config_path)


def test_missing_required_path_is_identified(tmp_path):
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text(
        "project_name: demo\n"
        "source_directories:\n"
        "  canonical: ./scripts\n"
        "storage_directory: ./data\n"
        "embedding_model: test-model\n"
        "ollama_url: http://127.0.0.1:11434\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="fanfiction"):
        load_config(config_path)


def test_multiple_canonical_paths_resolve_relative_to_config(tmp_path):
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text(
        "project_name: demo\n"
        "source_directories:\n"
        "  canonical:\n"
        "    - ./buffy\n"
        "    - ./angel\n"
        "  fanfiction: ./fanfiction\n"
        "storage_directory: ./data\n"
        "embedding_model: test-model\n"
        "ollama_url: http://127.0.0.1:11434\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.canonical_directories == (
        (tmp_path / "buffy").resolve(),
        (tmp_path / "angel").resolve(),
    )
    assert config.fanfiction_directory == (tmp_path / "fanfiction").resolve()


def test_legacy_single_canonical_path_remains_supported(tmp_path):
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text(
        "project_name: demo\n"
        "source_directories:\n"
        "  canonical: ./scripts\n"
        "  fanfiction: ./fanfiction\n"
        "storage_directory: ./data\n"
        "embedding_model: test-model\n"
        "ollama_url: http://127.0.0.1:11434\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.canonical_directories == ((tmp_path / "scripts").resolve(),)


@pytest.mark.parametrize("canonical", ["[]", "[./scripts, '']", "null"])
def test_empty_or_invalid_canonical_path_list_is_rejected(tmp_path, canonical):
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text(
        "project_name: demo\n"
        "source_directories:\n"
        "  canonical: {}\n"
        "  fanfiction: ./fanfiction\n"
        "storage_directory: ./data\n"
        "embedding_model: test-model\n"
        "ollama_url: http://127.0.0.1:11434\n".format(canonical),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="canonical"):
        load_config(config_path)
