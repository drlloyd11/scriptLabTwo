from pathlib import Path
from types import SimpleNamespace

import pytest

import scriptlab.cli as cli_module
from scriptlab.cli import build_parser, main
from scriptlab.config import ConfigError, load_config
from scriptlab.vectors import VectorIndexError


def test_help_lists_implemented_and_planned_commands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "init",
        "inventory",
        "preview",
        "ingest",
        "index",
        "search",
        "status",
        "parse",
        "inspect",
        "validate",
    ):
        assert command in output


def test_search_arguments_are_parsed():
    args = build_parser().parse_args(
        [
            "search",
            "a concealed secret",
            "--mode",
            "hybrid",
            "--source",
            "both",
            "--work-id",
            "work-123",
            "--limit",
            "5",
        ]
    )

    assert args.query == "a concealed secret"
    assert args.mode == "hybrid"
    assert args.source == "both"
    assert args.work_id == "work-123"
    assert args.limit == 5


def test_search_session_reuses_search_options_for_multiple_queries(monkeypatch, capsys):
    queries = iter(["first question", "-second question", "/quit"])
    calls = []
    monkeypatch.setattr("builtins.input", lambda prompt: next(queries))
    monkeypatch.setattr(cli_module, "load_config", lambda path: object())

    def fake_search(config, query, **kwargs):
        calls.append((query, kwargs))
        return {
            "results": [],
            "index_version": "test-version",
            "coverage": {"documents_ready": 1, "documents_total": 1},
        }

    monkeypatch.setattr(cli_module, "search_vector_chunks", fake_search)

    assert main(
        ["search-session", "--source", "fanfiction", "--work-id", "work-123", "--limit", "5"]
    ) == 0
    assert [query for query, _ in calls] == ["first question", "-second question"]
    assert all(
        options == {"source_class": "fanfiction", "work_id": "work-123", "limit": 5}
        for _, options in calls
    )
    assert capsys.readouterr().out.count("requested_mode=semantic; actual_mode=semantic") == 2


def test_search_session_reports_failure_and_continues_until_eof(monkeypatch, capsys):
    queries = iter(["first question", "second question"])
    calls = []

    def read_query(prompt):
        try:
            return next(queries)
        except StopIteration:
            raise EOFError

    def fake_search(config, query, **kwargs):
        calls.append(query)
        if len(calls) == 1:
            raise VectorIndexError("synthetic vector outage")
        return {
            "results": [],
            "index_version": "test-version",
            "coverage": {"documents_ready": 1, "documents_total": 1},
        }

    monkeypatch.setattr("builtins.input", read_query)
    monkeypatch.setattr(cli_module, "load_config", lambda path: object())
    monkeypatch.setattr(cli_module, "search_vector_chunks", fake_search)

    assert main(["search-session"]) == 1
    assert calls == ["first question", "second question"]
    output = capsys.readouterr()
    assert "synthetic vector outage" in output.err
    assert "requested_mode=semantic; actual_mode=semantic" in output.out


def test_search_session_reports_timings_for_each_query(monkeypatch, capsys):
    queries = iter(["first question", "second question", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(queries))
    monkeypatch.setattr(cli_module, "load_config", lambda path: object())

    def fake_search(config, query, **kwargs):
        assert kwargs["timings"] is not None
        return {
            "results": [],
            "index_version": "test-version",
            "coverage": {"documents_ready": 1, "documents_total": 1},
        }

    monkeypatch.setattr(cli_module, "search_vector_chunks", fake_search)
    assert main(["search-session", "--timings"]) == 0
    output = capsys.readouterr()
    assert output.err.count("Search timings") == 2
    assert output.err.count("retrieval.total:") == 2


def test_rrf_combines_ranked_results_and_deduplicates_chunks():
    fts_results = [{"chunk_id": "shared"}, {"chunk_id": "fts-only"}]
    vector_results = [{"chunk_id": "shared"}, {"chunk_id": "vector-only"}]

    results = cli_module._reciprocal_rank_fusion(fts_results, vector_results, 3)

    assert [result["chunk_id"] for result in results] == [
        "shared",
        "fts-only",
        "vector-only",
    ]
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_semantic_search_requires_explicit_fts_fallback(tmp_path, capsys, monkeypatch):
    scripts = tmp_path / "scripts"
    fanfiction = tmp_path / "fanfiction"
    scripts.mkdir()
    fanfiction.mkdir()
    (scripts / "episode.txt").write_text(
        "INT. LIBRARY - NIGHT\n\nBUFFY\nThe hidden book holds the secret.\n",
        encoding="utf-8",
    )
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
    command = ["--config", str(config_path)]
    assert main(command + ["ingest"]) == 0
    assert main(command + ["parse"]) == 0

    def unavailable(*args, **kwargs):
        raise VectorIndexError("synthetic vector outage")

    monkeypatch.setattr(cli_module, "search_vector_chunks", unavailable)
    assert main(command + ["search", "hidden book", "--mode", "semantic"]) == 2
    assert "synthetic vector outage" in capsys.readouterr().err

    assert main(
        command
        + [
            "search",
            "hidden book",
            "--mode",
            "semantic",
            "--fallback-to-fts",
            "--timings",
        ]
    ) == 0
    output = capsys.readouterr()
    assert "requested_mode=semantic; actual_mode=fts" in output.out
    assert "episode.txt [canonical]" in output.out
    assert "using FTS by explicit request" in output.err
    assert "actual_mode=fts" in output.err
    assert "vector.total:" in output.err
    assert "fts.sql:" in output.err

def test_index_arguments_support_batches_limits_and_rebuild():
    args = build_parser().parse_args(
        [
            "index",
            "--batch-size",
            "4",
            "--limit",
            "12",
            "--rebuild",
            "--retry-failed",
        ]
    )

    assert args.batch_size == 4
    assert args.limit == 12
    assert args.rebuild
    assert args.retry_failed


def test_index_command_displays_actual_incomplete_scan_progress(tmp_path, capsys, monkeypatch):
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

    def fake_index_chunks(config, **kwargs):
        kwargs["progress_callback"](25, 100, 25, 25, 0, 0)
        return SimpleNamespace(
            index_ready=False,
            embedding_model="test-model",
            chunks_indexed_this_run=25,
            chunks_reused=0,
            chunks_ready=25,
            chunks_expected=100,
            chunks_failed=0,
            documents_ready=0,
            errors=(),
        )

    monkeypatch.setattr(cli_module, "index_chunks", fake_index_chunks)
    monkeypatch.setattr(cli_module, "get_status", lambda config: {"documents": 1})

    assert main(["--config", str(config_path), "index"]) == 0

    output = capsys.readouterr().out
    assert "Index scan:  25.0% (25/100 checked)" in output
    assert "Index scan: 100.0%" not in output
    assert "Vector index building" in output


def test_fts_search_and_validate_commands(tmp_path, capsys):
    scripts = tmp_path / "scripts"
    fanfiction = tmp_path / "fanfiction"
    scripts.mkdir()
    fanfiction.mkdir()
    (scripts / "episode.txt").write_text(
        "INT. LIBRARY - NIGHT\n\nBUFFY\nThe hidden book holds the secret.\n",
        encoding="utf-8",
    )
    (fanfiction / "story.txt").write_text(
        "Chapter One\n\nWillow finds the hidden book and its secret.\n",
        encoding="utf-8",
    )
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
    command = ["--config", str(config_path)]

    assert main(command + ["ingest"]) == 0
    assert main(command + ["parse"]) == 0
    assert main(command + ["search", "hidden book secret"]) == 0
    output = capsys.readouterr().out
    assert "Search mode: fts; source: canonical; results:" in output
    assert "episode.txt [canonical]" in output
    assert "story.txt [fanfiction]" not in output
    assert "Source spans:" in output

    assert main(command + ["search", "hidden book secret", "--source", "both"]) == 0
    output = capsys.readouterr().out
    assert "episode.txt [canonical]" in output
    assert "story.txt [fanfiction]" in output

    assert main(command + ["validate"]) == 0
    output = capsys.readouterr().out
    assert "FTS integrity and chunk citations: valid" in output

    assert main(command + ["search", "secret", "--mode", "semantic"]) == 2
    assert "No complete active vector index" in capsys.readouterr().err


def test_init_writes_config_without_creating_source_directories(tmp_path):
    config_path = tmp_path / "scriptlab.yaml"

    assert main(["init", "--output", str(config_path)]) == 0
    config = load_config(config_path)

    assert config_path.is_file()
    assert config.canonical_directories == (
        (tmp_path.parent / "Buffy shooting scripts").resolve(),
        (tmp_path.parent / "angel_shooting_scripts").resolve(),
    )
    assert config.fanfiction_directory == (
        tmp_path.parent / "kitten_fanfics"
    ).resolve()
    assert all(not directory.exists() for directory in config.canonical_directories)
    assert not config.fanfiction_directory.exists()
    assert not config.storage_directory.exists()


def test_init_refuses_to_overwrite_existing_config(tmp_path, capsys):
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text("keep this file", encoding="utf-8")

    assert main(["init", "--output", str(config_path)]) == 2

    assert config_path.read_text(encoding="utf-8") == "keep this file"
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_command_reports_missing_config_clearly(tmp_path, capsys):
    missing_config = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit) as exc_info:
        main(["--config", str(missing_config), "status"])

    assert exc_info.value.code == 2
    assert "Configuration file not found" in capsys.readouterr().err


def test_index_requires_parsed_chunks(tmp_path, capsys):
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

    assert main(["--config", str(config_path), "index"]) == 2
    assert "No parsed chunks are available to index" in capsys.readouterr().err


def test_synthetic_fixtures_are_present_and_nonempty():
    fixture_directory = Path(__file__).parent / "fixtures"

    assert (fixture_directory / "sample_script.txt").read_text(
        encoding="utf-8"
    ).strip()
    assert (fixture_directory / "sample_fanfiction.html").read_text(
        encoding="utf-8"
    ).strip()


def test_inventory_and_preview_surface_txt_recovery_warnings(tmp_path, capsys):
    scripts = tmp_path / "scripts"
    fanfiction = tmp_path / "fanfiction"
    scripts.mkdir()
    fanfiction.mkdir()
    source = scripts / "recover.txt"
    raw = b"MARA\x00\nHello."
    source.write_bytes(raw)
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

    assert main(["--config", str(config_path), "inventory"]) == 0
    output = capsys.readouterr()
    assert "recovered: 1" in output.out
    assert "Recovery warning:" in output.err
    assert "U+0000=1" in output.err

    assert main(["--config", str(config_path), "preview", str(source)]) == 0
    output = capsys.readouterr()
    assert "MARA\nHello." in output.out
    assert "Recovery warning:" in output.err
    assert "U+0000=1" in output.err
    assert source.read_bytes() == raw


def test_ingest_and_status_report_saved_counts_and_conflicts(tmp_path, capsys):
    scripts = tmp_path / "scripts"
    fanfiction = tmp_path / "fanfiction"
    scripts.mkdir()
    fanfiction.mkdir()
    source = scripts / "episode.txt"
    source.write_text("Original.", encoding="utf-8")
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
    command = ["--config", str(config_path)]

    assert main(command + ["ingest"]) == 0
    assert "Imported: 1" in capsys.readouterr().out
    assert main(command + ["status"]) == 0
    assert "Works: 1; documents: 1" in capsys.readouterr().out

    source.write_text("Changed.", encoding="utf-8")
    assert main(command + ["ingest"]) == 1
    assert "conflicts: 1" in capsys.readouterr().out
    assert main(command + ["status"]) == 1
    assert "changed since import" in capsys.readouterr().err


def test_parse_and_inspect_commands_show_source_spans_and_warnings(tmp_path, capsys):
    scripts = tmp_path / "scripts"
    fanfiction = tmp_path / "fanfiction"
    scripts.mkdir()
    fanfiction.mkdir()
    source = scripts / "episode.txt"
    source.write_text(
        "INT. ROOM - DAY\n\nBUFFY\nWait.\n\nTHE DOOR IS OPEN\n"
        "She steps back.\n",
        encoding="utf-8",
    )
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
    command = ["--config", str(config_path)]
    assert main(command + ["ingest"]) == 0
    assert main(command + ["parse"]) == 0
    assert "warnings: 1" in capsys.readouterr().out

    import sqlite3

    with sqlite3.connect(tmp_path / "data" / "scriptlab.sqlite3") as connection:
        document_id = connection.execute(
            "SELECT document_id FROM documents WHERE filename = ?", (source.name,)
        ).fetchone()[0]
    assert main(command + ["inspect", document_id]) == 0
    output = capsys.readouterr().out
    assert "Scene 0" in output
    assert "speaker=BUFFY resolved=unresolved" in output
    assert "Span 0" in output
    assert "Warning ambiguous_speaker" in output
    assert "THE DOOR IS OPEN" in output
