import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from scriptlab.config import ProjectConfig
from scriptlab.corpus import FlatFolderError
from scriptlab.database import (
    DATABASE_FILENAME,
    PersistenceError,
    _normalized_absolute_path,
    get_status,
    initialize_database,
    import_sources,
    parse_documents,
    search_chunks,
    validate_database,
)
from scriptlab.parser import COMPOSITE_SEPARATOR


def make_config(
    base,
    canonical_directories=None,
    fanfiction_directory=None,
    storage_directory=None,
):
    canonical = tuple(canonical_directories or (base / "scripts",))
    fanfiction = fanfiction_directory or base / "fanfiction"
    storage = storage_directory or base / "data"
    for directory in canonical + (fanfiction,):
        directory.mkdir(parents=True, exist_ok=True)
    return ProjectConfig(
        project_name="test",
        canonical_directories=canonical,
        fanfiction_directory=fanfiction,
        storage_directory=storage,
        embedding_model="test-model",
        ollama_url="http://127.0.0.1:11434",
    )


def rows(database_path, query, parameters=()):
    with sqlite3.connect(database_path) as connection:
        return connection.execute(query, parameters).fetchall()


def create_populated_v1_database(config):
    config.storage_directory.mkdir(parents=True, exist_ok=True)
    database_path = config.storage_directory / DATABASE_FILENAME
    canonical_text = "INT. ROOM - DAY\nBUFFY\nHello."
    fanfiction_text = 'Chapter One\n\n"Hello," Willow said.'
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE works (
                work_id TEXT PRIMARY KEY,
                provisional_title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                work_id TEXT NOT NULL UNIQUE REFERENCES works(work_id),
                source_class TEXT NOT NULL CHECK (source_class IN ('canonical', 'fanfiction')),
                source_path TEXT NOT NULL,
                normalized_absolute_path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                raw_sha256 TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                recovery_warnings TEXT NOT NULL,
                import_state TEXT NOT NULL CHECK (import_state = 'ready'),
                imported_at TEXT NOT NULL
            );
            CREATE TABLE scenes (
                scene_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id),
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                scene_text TEXT NOT NULL
            );
            CREATE TABLE dialogue_turns (
                turn_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id),
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                turn_text TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id),
                chunk_type TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                chunk_text TEXT NOT NULL
            );
            CREATE TABLE import_issues (
                normalized_absolute_path TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_class TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )
        for document_id, work_id, source_class, filename, text in (
            ("doc-canonical", "work-canonical", "canonical", "episode.txt", canonical_text),
            ("doc-fanfiction", "work-fanfiction", "fanfiction", "story.txt", fanfiction_text),
        ):
            source_path = config.canonical_directories[0] / filename
            connection.execute(
                "INSERT INTO works VALUES (?, ?, ?)",
                (work_id, Path(filename).stem, "2026-01-01T00:00:00+00:00"),
            )
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document_id,
                    work_id,
                    source_class,
                    str(source_path),
                    str(source_path).casefold(),
                    filename,
                    "a" * 64,
                    text,
                    "[]",
                    "ready",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
        connection.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            (
                "legacy-chunk",
                "doc-canonical",
                "scene",
                0,
                len(canonical_text),
                canonical_text,
            ),
        )
    return database_path


def test_import_persists_full_normalized_text_and_is_idempotent(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "episode.txt"
    raw = b"INT. ROOM - DAY\r\n\r\nMARA\r\nHello."
    source.write_bytes(raw)

    first = import_sources(config)
    first_ids = rows(
        config.storage_directory / DATABASE_FILENAME,
        "SELECT work_id, document_id FROM documents",
    )
    second = import_sources(config)
    database_path = config.storage_directory / DATABASE_FILENAME

    assert first.added == 1
    assert second.added == 0
    assert second.unchanged == 1
    assert first_ids == rows(
        database_path, "SELECT work_id, document_id FROM documents"
    )
    assert rows(database_path, "SELECT COUNT(*) FROM works") == [(1,)]
    assert rows(database_path, "SELECT COUNT(*) FROM documents") == [(1,)]
    document = rows(
        database_path,
        "SELECT source_class, filename, raw_sha256, normalized_text, import_state "
        "FROM documents",
    )[0]
    assert document == (
        "canonical",
        "episode.txt",
        hashlib.sha256(raw).hexdigest(),
        "INT. ROOM - DAY\n\nMARA\nHello.",
        "ready",
    )
    assert rows(database_path, "SELECT COUNT(*) FROM scenes") == [(0,)]
    assert rows(database_path, "SELECT COUNT(*) FROM dialogue_turns") == [(0,)]
    assert rows(database_path, "SELECT COUNT(*) FROM chunks") == [(0,)]


def test_changed_bytes_are_reported_without_overwriting_existing_document(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "episode.txt"
    source.write_text("Original text.", encoding="utf-8")
    import_sources(config)
    source.write_text("Changed text.", encoding="utf-8")

    result = import_sources(config)
    status = get_status(config)
    database_path = config.storage_directory / DATABASE_FILENAME
    document = rows(
        database_path,
        "SELECT raw_sha256, normalized_text FROM documents",
    )[0]

    assert result.conflicts == 1
    assert document == (
        hashlib.sha256(b"Original text.").hexdigest(),
        "Original text.",
    )
    assert status["issues"][0]["status"] == "conflict"
    assert "changed since import" in status["issues"][0]["message"]


def test_same_filename_in_different_directories_is_distinct(tmp_path):
    first_root = tmp_path / "scripts-a"
    second_root = tmp_path / "scripts-b"
    config = make_config(
        tmp_path, canonical_directories=(first_root, second_root)
    )
    (first_root / "episode.txt").write_text("First.", encoding="utf-8")
    (second_root / "episode.txt").write_text("Second.", encoding="utf-8")

    result = import_sources(config)
    database_path = config.storage_directory / DATABASE_FILENAME

    assert result.added == 2
    assert rows(database_path, "SELECT COUNT(*) FROM works") == [(2,)]
    assert rows(database_path, "SELECT COUNT(DISTINCT work_id) FROM documents") == [(2,)]


def test_normalized_path_is_case_insensitive_and_unicode_nfc():
    first = _normalized_absolute_path(Path("C:/archive/\u00c9pisode.txt"))
    second = _normalized_absolute_path(Path("c:/ARCHIVE/E\u0301PISODE.TXT"))

    assert first == second


def test_source_class_change_is_a_conflict_not_a_second_document(tmp_path):
    original_root = tmp_path / "original"
    other_root = tmp_path / "other"
    initial_config = make_config(
        tmp_path,
        canonical_directories=(original_root,),
        fanfiction_directory=other_root,
    )
    source = original_root / "episode.txt"
    source.write_text("Same text.", encoding="utf-8")
    import_sources(initial_config)

    changed_config = make_config(
        tmp_path,
        canonical_directories=(other_root,),
        fanfiction_directory=original_root,
        storage_directory=initial_config.storage_directory,
    )
    result = import_sources(changed_config)
    database_path = initial_config.storage_directory / DATABASE_FILENAME

    assert result.conflicts == 1
    assert rows(database_path, "SELECT COUNT(*) FROM documents") == [(1,)]
    assert rows(database_path, "SELECT source_class FROM documents") == [("canonical",)]
    assert get_status(changed_config)["issues"][0]["status"] == "conflict"


def test_unsafe_source_layout_fails_before_database_creation(tmp_path):
    config = make_config(tmp_path)
    (config.canonical_directories[0] / "nested").mkdir()

    with pytest.raises(FlatFolderError, match="Nested directories"):
        import_sources(config)

    assert not config.storage_directory.exists()


def test_unsupported_files_are_visible_in_status(tmp_path):
    config = make_config(tmp_path)
    (config.canonical_directories[0] / "notes.pdf").write_bytes(b"not opened")

    result = import_sources(config)
    issue = get_status(config)["issues"][0]

    assert result.unsupported == 1
    assert issue["status"] == "unsupported"
    assert "pdf" in issue["message"]


def test_populated_v1_database_migrates_to_v5_without_losing_corpus(tmp_path):
    config = make_config(tmp_path)
    database_path = create_populated_v1_database(config)
    original_documents = rows(
        database_path,
        "SELECT document_id, source_class, source_path, raw_sha256, normalized_text "
        "FROM documents ORDER BY document_id",
    )

    initialize_database(config)
    initialize_database(config)

    assert rows(database_path, "PRAGMA user_version") == [(5,)]
    assert rows(
        database_path,
        "SELECT document_id, source_class, source_path, raw_sha256, normalized_text "
        "FROM documents ORDER BY document_id",
    ) == original_documents
    assert rows(
        database_path,
        "SELECT chunk_id, chunker_version, chunk_ordinal FROM chunks",
    ) == [("legacy-chunk", "legacy_v1", 0)]
    assert rows(
        database_path,
        "SELECT chunk_id, span_ordinal, start_offset, end_offset FROM chunk_spans",
    ) == [("legacy-chunk", 0, 0, len(original_documents[0][4]))]
    assert rows(
        database_path,
        "SELECT chunk_id FROM chunk_fts WHERE chunk_fts MATCH 'BUFFY'",
    ) == [("legacy-chunk",)]
    assert rows(
        database_path,
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name IN ('vector_indexes', 'embedding_jobs') ORDER BY name",
    ) == [("embedding_jobs",), ("vector_indexes",)]
    assert rows(
        database_path,
        "SELECT DISTINCT layout_spans_json FROM documents",
    ) == [("[]",)]
    assert get_status(config)["documents"] == 2
    assert get_status(config)["vector_index"]["status"] == "not_built"


def test_import_persists_html_layout_and_parser_uses_it(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "layout_episode.htm"
    source.write_text(
        "<html><body><p>INT. ROOM - DAY</p>"
        "<p>THE DOOR</p><p>Swings open.</p>"
        "<center><font face='Verdana'>XANDER<br>Look behind you.</font></center>"
        "<p align='center' style='text-align:center'><font face='Verdana'>"
        "ANGEL<br>Watch behind you.</font></p>"
        "<div align='right'>CAMERA PANS</div>"
        "<table><tr><td bgcolor='#6666ff'><font size='+2'>ACT ONE</font>"
        "</td></tr></table>"
        "<center>END OF ACT ONE</center>"
        "<center><font face='Verdana'>BUFFY<br>I see it.</font></center>"
        "</body></html>",
        encoding="utf-8",
    )

    imported = import_sources(config)
    assert imported.added == 1
    database_path = config.storage_directory / DATABASE_FILENAME
    layout_json = rows(
        database_path,
        "SELECT layout_spans_json FROM documents WHERE filename = ?",
        (source.name,),
    )[0][0]
    layout_spans = json.loads(layout_json)
    assert {span["role"] for span in layout_spans} == {
        "center",
        "right_aligned",
        "styled_heading",
        "act_marker",
    }

    summary = parse_documents(config)
    assert not summary.errors
    document_id = rows(
        database_path,
        "SELECT document_id FROM documents WHERE filename = ?",
        (source.name,),
    )[0][0]
    turns = rows(
        database_path,
        "SELECT original_speaker_label, turn_text FROM dialogue_turns "
        "WHERE document_id = ? ORDER BY turn_ordinal",
        (document_id,),
    )
    assert [turn[0] for turn in turns] == [
        "XANDER",
        "ANGEL",
        "BUFFY",
    ]
    assert [turn[1] for turn in turns] == [
        "XANDER\nLook behind you.",
        "ANGEL\nWatch behind you.",
        "BUFFY\nI see it.",
    ]
    normalized_text = rows(
        database_path,
        "SELECT normalized_text FROM documents WHERE document_id = ?",
        (document_id,),
    )[0][0]
    assert "THE DOOR\n\nSwings open." in normalized_text


def test_failed_v1_migration_rolls_back_schema_and_data(tmp_path):
    config = make_config(tmp_path)
    database_path = create_populated_v1_database(config)
    original_documents = rows(
        database_path,
        "SELECT document_id, raw_sha256, normalized_text FROM documents "
        "ORDER BY document_id",
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE chunks SET end_offset = end_offset + 1 WHERE chunk_id = ?",
            ("legacy-chunk",),
        )

    with pytest.raises(PersistenceError, match="outside its document"):
        initialize_database(config)

    assert rows(database_path, "PRAGMA user_version") == [(1,)]
    assert rows(
        database_path,
        "SELECT document_id, raw_sha256, normalized_text FROM documents "
        "ORDER BY document_id",
    ) == original_documents
    assert "chunker_version" not in {
        row[1] for row in rows(database_path, "PRAGMA table_info(chunks)")
    }
    assert rows(
        database_path,
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('chunk_spans', 'parse_warnings')",
    ) == []


def test_fts_search_filters_sources_and_work_and_returns_citations(tmp_path):
    config = make_config(tmp_path)
    canonical = config.canonical_directories[0] / "episode.txt"
    fanfiction = config.fanfiction_directory / "story.txt"
    canonical.write_text(
        "INT. LIBRARY - NIGHT\n\nBUFFY\nThe hidden book holds the secret.\n",
        encoding="utf-8",
    )
    fanfiction.write_text(
        "Chapter One\n\nWillow finds the hidden book and its secret.\n",
        encoding="utf-8",
    )
    import_sources(config)
    summary = parse_documents(config)
    assert not summary.errors

    canonical_results = search_chunks(config, "hidden book secret")
    assert canonical_results
    assert {result["source_class"] for result in canonical_results} == {"canonical"}
    assert any("BUFFY" in result["speaker_labels"] for result in canonical_results)
    assert all(result["spans"] for result in canonical_results)

    both_results = search_chunks(config, "hidden book secret", source_class="both")
    assert {result["source_class"] for result in both_results} == {
        "canonical",
        "fanfiction",
    }
    canonical_work_id = next(
        result["work_id"]
        for result in both_results
        if result["source_class"] == "canonical"
    )
    work_results = search_chunks(
        config, "hidden book secret", source_class="both", work_id=canonical_work_id
    )
    assert {result["work_id"] for result in work_results} == {canonical_work_id}

    with sqlite3.connect(config.storage_directory / DATABASE_FILENAME) as connection:
        source_text = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT document_id, normalized_text FROM documents"
            )
        }
        chunk_rows = connection.execute(
            "SELECT chunk_id, chunk_text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        fts_rows = connection.execute(
            "SELECT chunk_id, chunk_text FROM chunk_fts ORDER BY chunk_id"
        ).fetchall()
    assert fts_rows == chunk_rows
    for result in both_results:
        reconstructed = COMPOSITE_SEPARATOR.join(
            source_text[result["document_id"]][span["start_offset"] : span["end_offset"]]
            for span in result["spans"]
        )
        assert reconstructed == result["chunk_text"]

    validation = validate_database(config)
    assert validation["valid"]
    assert validation["fts_rows"] == validation["chunks"]

    original_result_ids = {result["chunk_id"] for result in both_results}
    replacement = parse_documents(config, chunker_version="test-chunker-v2")
    assert not replacement.errors
    updated_results = search_chunks(config, "hidden book secret", source_class="both")
    assert {result["chunk_id"] for result in updated_results} != original_result_ids
    validation = validate_database(config)
    assert validation["valid"]
    assert validation["fts_rows"] == validation["chunks"]

def test_search_chunks_can_expand_hybrid_candidates_with_or_matching(tmp_path):
    config = make_config(tmp_path)
    (config.canonical_directories[0] / "episode.txt").write_text(
        "INT. ROOM - DAY\n\nBUFFY\nThe hidden book holds the secret.\n",
        encoding="utf-8",
    )
    import_sources(config)
    summary = parse_documents(config)
    assert not summary.errors

    strict_results = search_chunks(config, "hidden nonexistent")
    hybrid_candidates = search_chunks(
        config, "hidden nonexistent", require_all_terms=False
    )

    assert not strict_results
    assert hybrid_candidates
    assert {result["source_class"] for result in hybrid_candidates} == {"canonical"}