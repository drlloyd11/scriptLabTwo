"""SQLite persistence for the ScriptLab corpus."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import unicodedata
import uuid
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from scriptlab.config import ProjectConfig
from scriptlab.corpus import (
    CorpusError,
    SourceRoot,
    _assert_path_has_no_reparse_components,
    _read_and_extract,
    _read_and_extract_with_layout,
    _validate_report_location,
    scan_sources,
)
from scriptlab.parser import (
    COMPOSITE_SEPARATOR,
    ParseResult,
    parse_document,
    validate_parse_result,
)
from scriptlab.timing import SearchTimings, measure


DATABASE_FILENAME = "scriptlab.sqlite3"
_SCHEMA_VERSION = 6
_ID_NAMESPACE = uuid.UUID("3c1dc0d2-f3a9-4c0f-b980-c1bd7cd12489")
_READY_DOCUMENT_SQL = """
    EXISTS (
        SELECT 1 FROM document_chunk_counts AS dc
        JOIN document_index_ready_counts AS dr
          ON dr.document_id = dc.document_id AND dr.index_version = ?
        WHERE dc.document_id = d.document_id
          AND dc.chunk_count > 0 AND dr.ready_count = dc.chunk_count
    )
"""


class PersistenceError(RuntimeError):
    """Raised when the ScriptLab database cannot be initialized or used."""


@dataclass(frozen=True)
class ImportResult:
    added: int
    unchanged: int
    conflicts: int
    failed: int
    unsupported: int


@dataclass(frozen=True)
class ParseSummary:
    documents_parsed: int
    scenes: int
    dialogue_turns: int
    chunks: int
    warnings: int
    errors: Tuple[str, ...]


def _configured_roots(config: ProjectConfig) -> Tuple[SourceRoot, ...]:
    return tuple(
        SourceRoot(path=path, source_class="canonical")
        for path in config.canonical_directories
    ) + (SourceRoot(path=config.fanfiction_directory, source_class="fanfiction"),)


def _normalized_absolute_path(path: Path) -> str:
    absolute_path = os.path.abspath(os.fspath(path))
    normalized = unicodedata.normalize("NFC", os.path.normcase(absolute_path))
    return normalized.casefold()


def _stable_id(kind: str, normalized_path: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, "scriptlab:{}:{}".format(kind, normalized_path)))


def _database_path(config: ProjectConfig, roots: Tuple[SourceRoot, ...]) -> Path:
    _validate_report_location(config, roots)
    storage = config.storage_directory
    try:
        storage.mkdir(parents=True, exist_ok=True)
        storage_info = _assert_path_has_no_reparse_components(
            storage, allow_missing_leaf=True
        )
        if storage_info is None or not stat.S_ISDIR(storage_info.st_mode):
            raise CorpusError("Storage directory could not be created: {!s}".format(storage))
        path = storage / DATABASE_FILENAME
        database_info = _assert_path_has_no_reparse_components(
            path, allow_missing_leaf=True
        )
        if database_info is not None and not path.is_file():
            raise CorpusError("Database path is not a regular file: {!s}".format(path))
    except (OSError, CorpusError) as exc:
        raise PersistenceError("Unsafe or unavailable database path: {}".format(exc)) from exc
    return path


@contextmanager
def _connection(config: ProjectConfig, roots: Tuple[SourceRoot, ...]) -> Iterator[sqlite3.Connection]:
    path = _database_path(config, roots)
    connection = None
    try:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        _initialize_schema(connection)
        yield connection
    except sqlite3.Error as exc:
        raise PersistenceError("Could not use database {!s}: {}".format(path, exc)) from exc
    finally:
        if connection is not None:
            connection.close()


def _initialize_schema(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version > _SCHEMA_VERSION:
        raise PersistenceError(
            "Database schema version {} is newer than supported version {}.".format(
                version, _SCHEMA_VERSION
            )
        )
    if version == 0:
        _create_v1_schema(connection)
        version = 1
    if version == 1:
        _migrate_v1_to_v2(connection)
        version = 2
    if version == 2:
        _migrate_v2_to_v3(connection)
        version = 3
    if version == 3:
        _migrate_v3_to_v4(connection)
        version = 4
    if version == 4:
        _migrate_v4_to_v5(connection)
        version = 5
    if version == 5:
        _migrate_v5_to_v6(connection)
        version = 6
    if version != _SCHEMA_VERSION:
        raise PersistenceError(
            "Unsupported database schema version {}.".format(version)
        )


def _create_v1_schema(connection: sqlite3.Connection) -> None:
    statements = (
        "CREATE TABLE works (work_id TEXT PRIMARY KEY, provisional_title TEXT NOT NULL, created_at TEXT NOT NULL)",
        "CREATE TABLE documents (document_id TEXT PRIMARY KEY, work_id TEXT NOT NULL UNIQUE REFERENCES works(work_id), source_class TEXT NOT NULL CHECK (source_class IN ('canonical', 'fanfiction')), source_path TEXT NOT NULL, normalized_absolute_path TEXT NOT NULL UNIQUE, filename TEXT NOT NULL, raw_sha256 TEXT NOT NULL, normalized_text TEXT NOT NULL, recovery_warnings TEXT NOT NULL, import_state TEXT NOT NULL CHECK (import_state = 'ready'), imported_at TEXT NOT NULL)",
        "CREATE TABLE scenes (scene_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id), start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL, scene_text TEXT NOT NULL)",
        "CREATE TABLE dialogue_turns (turn_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id), start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL, turn_text TEXT NOT NULL)",
        "CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id), chunk_type TEXT NOT NULL, start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL, chunk_text TEXT NOT NULL)",
        "CREATE TABLE import_issues (normalized_absolute_path TEXT PRIMARY KEY, source_path TEXT NOT NULL, source_class TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL, updated_at TEXT NOT NULL)",
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        for statement in statements:
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("ALTER TABLE scenes ADD COLUMN scene_heading TEXT")
        connection.execute(
            "ALTER TABLE scenes ADD COLUMN scene_ordinal INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            "ALTER TABLE scenes ADD COLUMN scene_type TEXT NOT NULL DEFAULT 'unknown'"
        )
        connection.execute(
            "ALTER TABLE dialogue_turns ADD COLUMN scene_id TEXT "
            "REFERENCES scenes(scene_id)"
        )
        connection.execute(
            "ALTER TABLE dialogue_turns ADD COLUMN original_speaker_label TEXT"
        )
        connection.execute(
            "ALTER TABLE dialogue_turns ADD COLUMN speaker_character_id TEXT"
        )
        connection.execute(
            "ALTER TABLE dialogue_turns ADD COLUMN attribution_status TEXT "
            "NOT NULL DEFAULT 'unknown'"
        )
        connection.execute(
            "ALTER TABLE dialogue_turns ADD COLUMN turn_ordinal INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            "ALTER TABLE chunks ADD COLUMN chunker_version TEXT "
            "NOT NULL DEFAULT 'legacy_v1'"
        )
        connection.execute(
            "ALTER TABLE chunks ADD COLUMN chunk_ordinal INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            """
            CREATE TABLE chunk_spans (
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                span_ordinal INTEGER NOT NULL CHECK (span_ordinal >= 0),
                start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
                end_offset INTEGER NOT NULL CHECK (end_offset > start_offset),
                PRIMARY KEY (chunk_id, span_ordinal)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE parse_warnings (
                warning_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                scene_id TEXT REFERENCES scenes(scene_id) ON DELETE CASCADE,
                start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
                end_offset INTEGER NOT NULL CHECK (end_offset > start_offset),
                warning_code TEXT NOT NULL,
                message TEXT NOT NULL,
                parser_version TEXT NOT NULL
            )
            """
        )

        chunks = connection.execute(
            """
            SELECT c.chunk_id, c.document_id, c.start_offset, c.end_offset,
                   c.chunk_text, d.normalized_text
            FROM chunks AS c
            JOIN documents AS d ON d.document_id = c.document_id
            ORDER BY c.document_id, c.start_offset, c.end_offset, c.chunk_id
            """
        ).fetchall()
        ordinals = {}
        for chunk in chunks:
            document_id = chunk["document_id"]
            ordinal = ordinals.get(document_id, 0)
            ordinals[document_id] = ordinal + 1
            text = chunk["normalized_text"]
            start_offset = chunk["start_offset"]
            end_offset = chunk["end_offset"]
            if not (0 <= start_offset < end_offset <= len(text)):
                raise PersistenceError(
                    "Cannot migrate legacy chunk {!r}: offsets are outside its document.".format(
                        chunk["chunk_id"]
                    )
                )
            if text[start_offset:end_offset] != chunk["chunk_text"]:
                raise PersistenceError(
                    "Cannot migrate legacy chunk {!r}: stored text does not match its source span.".format(
                        chunk["chunk_id"]
                    )
                )
            connection.execute(
                "UPDATE chunks SET chunk_ordinal = ? WHERE chunk_id = ?",
                (ordinal, chunk["chunk_id"]),
            )
            connection.execute(
                "INSERT INTO chunk_spans "
                "(chunk_id, span_ordinal, start_offset, end_offset) "
                "VALUES (?, 0, ?, ?)",
                (chunk["chunk_id"], start_offset, end_offset),
            )

        scenes = connection.execute(
            "SELECT scene_id, document_id FROM scenes "
            "ORDER BY document_id, start_offset, end_offset, scene_id"
        ).fetchall()
        scene_ordinals = {}
        for scene in scenes:
            ordinal = scene_ordinals.get(scene["document_id"], 0)
            scene_ordinals[scene["document_id"]] = ordinal + 1
            connection.execute(
                "UPDATE scenes SET scene_ordinal = ? WHERE scene_id = ?",
                (ordinal, scene["scene_id"]),
            )

        turns = connection.execute(
            "SELECT turn_id, document_id FROM dialogue_turns "
            "ORDER BY document_id, start_offset, end_offset, turn_id"
        ).fetchall()
        turn_ordinals = {}
        for turn in turns:
            ordinal = turn_ordinals.get(turn["document_id"], 0)
            turn_ordinals[turn["document_id"]] = ordinal + 1
            connection.execute(
                "UPDATE dialogue_turns SET turn_ordinal = ? WHERE turn_id = ?",
                (ordinal, turn["turn_id"]),
            )

        connection.execute(
            "CREATE UNIQUE INDEX idx_chunks_document_version_ordinal "
            "ON chunks(document_id, chunker_version, chunk_ordinal)"
        )
        connection.execute(
            "CREATE INDEX idx_chunk_spans_chunk ON chunk_spans(chunk_id, span_ordinal)"
        )
        connection.execute(
            "CREATE INDEX idx_parse_warnings_document "
            "ON parse_warnings(document_id, start_offset, end_offset)"
        )
        connection.execute("PRAGMA user_version = 2")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE chunk_fts USING fts5(" 
            "chunk_id UNINDEXED, chunk_text, content='chunks', "
            "content_rowid='rowid')"
        )
        connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild')")
        for statement in (
            """
            CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunk_fts(rowid, chunk_id, chunk_text)
                VALUES (new.rowid, new.chunk_id, new.chunk_text);
            END
            """,
            """
            CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO chunk_fts(chunk_fts, rowid, chunk_id, chunk_text)
                VALUES ('delete', old.rowid, old.chunk_id, old.chunk_text);
            END
            """,
            """
            CREATE TRIGGER chunks_fts_update AFTER UPDATE OF chunk_id, chunk_text ON chunks BEGIN
                INSERT INTO chunk_fts(chunk_fts, rowid, chunk_id, chunk_text)
                VALUES ('delete', old.rowid, old.chunk_id, old.chunk_text);
                INSERT INTO chunk_fts(rowid, chunk_id, chunk_text)
                VALUES (new.rowid, new.chunk_id, new.chunk_text);
            END
            """,
        ):
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            CREATE TABLE vector_indexes (
                index_version TEXT PRIMARY KEY,
                embedding_model TEXT NOT NULL,
                collection_name TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL CHECK (status IN ('building', 'active', 'retired')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE UNIQUE INDEX idx_vector_indexes_single_active "
            "ON vector_indexes(status) WHERE status = 'active'"
        )
        connection.execute(
            """
            CREATE TABLE embedding_jobs (
                chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                index_version TEXT NOT NULL REFERENCES vector_indexes(index_version) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK (status IN ('ready', 'failed')),
                error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (chunk_id, index_version)
            )
            """
        )
        connection.execute(
            "CREATE INDEX idx_embedding_jobs_index_status "
            "ON embedding_jobs(index_version, status, chunk_id)"
        )
        connection.execute("PRAGMA user_version = 4")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _migrate_v4_to_v5(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "ALTER TABLE documents ADD COLUMN layout_spans_json TEXT "
            "NOT NULL DEFAULT '[]'"
        )
        connection.execute("PRAGMA user_version = 5")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
    connection.execute("BEGIN IMMEDIATE")
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == 6:
            connection.commit()
            return
        if version != 5:
            raise PersistenceError(
                "Cannot migrate readiness counts from schema version {}.".format(version)
            )
        connection.execute(
            """
            CREATE TABLE document_chunk_counts (
                document_id TEXT NOT NULL PRIMARY KEY
                    REFERENCES documents(document_id) ON DELETE CASCADE,
                chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE document_index_ready_counts (
                index_version TEXT NOT NULL
                    REFERENCES vector_indexes(index_version) ON DELETE CASCADE,
                document_id TEXT NOT NULL
                    REFERENCES documents(document_id) ON DELETE CASCADE,
                ready_count INTEGER NOT NULL CHECK (ready_count >= 0),
                PRIMARY KEY (index_version, document_id)
            )
            """
        )
        connection.execute(
            "INSERT INTO document_chunk_counts "
            "SELECT document_id, COUNT(*) FROM chunks GROUP BY document_id"
        )
        connection.execute(
            """
            INSERT INTO document_index_ready_counts
            SELECT j.index_version, c.document_id, COUNT(*)
            FROM embedding_jobs AS j
            JOIN chunks AS c ON c.chunk_id = j.chunk_id
            WHERE j.status = 'ready'
            GROUP BY j.index_version, c.document_id
            """
        )
        remove_chunk = """
            UPDATE document_index_ready_counts SET ready_count = ready_count - 1
            WHERE document_id = OLD.document_id AND index_version IN (
                SELECT index_version FROM embedding_jobs
                WHERE chunk_id = OLD.chunk_id AND status = 'ready'
            );
            UPDATE document_chunk_counts SET chunk_count = chunk_count - 1
            WHERE document_id = OLD.document_id;
        """
        add_chunk = """
            INSERT INTO document_chunk_counts(document_id, chunk_count)
            VALUES (NEW.document_id, 1)
            ON CONFLICT(document_id) DO UPDATE SET chunk_count = chunk_count + 1;
            INSERT INTO document_index_ready_counts(index_version, document_id, ready_count)
            SELECT index_version, NEW.document_id, 1 FROM embedding_jobs
            WHERE chunk_id = NEW.chunk_id AND status = 'ready'
            ON CONFLICT(index_version, document_id)
            DO UPDATE SET ready_count = ready_count + 1;
        """
        for statement in (
            "CREATE TRIGGER chunks_readiness_insert AFTER INSERT ON chunks BEGIN {} END".format(add_chunk),
            "CREATE TRIGGER chunks_readiness_delete BEFORE DELETE ON chunks BEGIN {} END".format(remove_chunk),
            "CREATE TRIGGER chunks_readiness_update_before BEFORE UPDATE OF chunk_id, document_id ON chunks "
            "WHEN OLD.chunk_id IS NOT NEW.chunk_id OR OLD.document_id IS NOT NEW.document_id "
            "BEGIN {} END".format(remove_chunk),
            "CREATE TRIGGER chunks_readiness_update_after AFTER UPDATE OF chunk_id, document_id ON chunks "
            "WHEN OLD.chunk_id IS NOT NEW.chunk_id OR OLD.document_id IS NOT NEW.document_id "
            "BEGIN {} END".format(add_chunk),
            """
            CREATE TRIGGER jobs_readiness_insert AFTER INSERT ON embedding_jobs
            WHEN NEW.status = 'ready' BEGIN
                INSERT INTO document_index_ready_counts(index_version, document_id, ready_count)
                SELECT NEW.index_version, document_id, 1 FROM chunks
                WHERE chunk_id = NEW.chunk_id
                ON CONFLICT(index_version, document_id)
                DO UPDATE SET ready_count = ready_count + 1;
            END
            """,
            """
            CREATE TRIGGER jobs_readiness_delete AFTER DELETE ON embedding_jobs
            WHEN OLD.status = 'ready' BEGIN
                UPDATE document_index_ready_counts SET ready_count = ready_count - 1
                WHERE index_version = OLD.index_version
                  AND document_id = (SELECT document_id FROM chunks WHERE chunk_id = OLD.chunk_id);
            END
            """,
            """
            CREATE TRIGGER jobs_readiness_update AFTER UPDATE OF chunk_id, index_version, status ON embedding_jobs
            BEGIN
                UPDATE document_index_ready_counts SET ready_count = ready_count - 1
                WHERE OLD.status = 'ready' AND index_version = OLD.index_version
                  AND document_id = (SELECT document_id FROM chunks WHERE chunk_id = OLD.chunk_id);
                INSERT INTO document_index_ready_counts(index_version, document_id, ready_count)
                SELECT NEW.index_version, document_id, 1 FROM chunks
                WHERE NEW.status = 'ready' AND chunk_id = NEW.chunk_id
                ON CONFLICT(index_version, document_id)
                DO UPDATE SET ready_count = ready_count + 1;
            END
            """,
        ):
            connection.execute(statement)
        connection.execute("PRAGMA user_version = 6")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def initialize_database(config: ProjectConfig) -> Path:
    """Create the ScriptLab database and its initial schema if needed."""
    roots = _configured_roots(config)
    with _connection(config, roots):
        pass
    return config.storage_directory / DATABASE_FILENAME


def _record_issue(
    connection: sqlite3.Connection,
    normalized_path: str,
    source_path: str,
    source_class: str,
    status: str,
    message: str,
) -> None:
    with connection:
        connection.execute(
            """
            INSERT INTO import_issues (
                normalized_absolute_path, source_path, source_class, status,
                message, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(normalized_absolute_path) DO UPDATE SET
                source_path = excluded.source_path,
                source_class = excluded.source_class,
                status = excluded.status,
                message = excluded.message,
                updated_at = excluded.updated_at
            """,
            (
                normalized_path,
                source_path,
                source_class,
                status,
                message,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def import_sources(config: ProjectConfig) -> ImportResult:
    """Extract configured sources and idempotently import them into SQLite."""
    scan = scan_sources(config)
    roots = scan.roots
    added = unchanged = conflicts = failed = 0
    with _connection(config, roots) as connection:
        for source_file in scan.files:
            path_key = _normalized_absolute_path(source_file.path)
            try:
                digest, _, text, recovery_warnings, layout_spans = _read_and_extract_with_layout(
                    source_file.path, source_file.file_format
                )
            except CorpusError as exc:
                _record_issue(
                    connection,
                    path_key,
                    str(source_file.path),
                    source_file.source_class,
                    "failed",
                    str(exc),
                )
                failed += 1
                continue

            normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
            layout_spans_json = json.dumps(
                list(layout_spans), ensure_ascii=False
            )
            existing = connection.execute(
                "SELECT source_class, raw_sha256, layout_spans_json FROM documents "
                "WHERE normalized_absolute_path = ?",
                (path_key,),
            ).fetchone()
            if existing is not None:
                if existing["source_class"] != source_file.source_class:
                    message = (
                        "Source class changed from {!r} to {!r}; existing document "
                        "was not modified.".format(
                            existing["source_class"], source_file.source_class
                        )
                    )
                    _record_issue(
                        connection,
                        path_key,
                        str(source_file.path),
                        source_file.source_class,
                        "conflict",
                        message,
                    )
                    conflicts += 1
                elif existing["raw_sha256"] != digest:
                    _record_issue(
                        connection,
                        path_key,
                        str(source_file.path),
                        source_file.source_class,
                        "conflict",
                        "Source bytes changed since import; existing document was not modified.",
                    )
                    conflicts += 1
                else:
                    with connection:
                        if existing["layout_spans_json"] != layout_spans_json:
                            connection.execute(
                                "UPDATE documents SET layout_spans_json = ? "
                                "WHERE normalized_absolute_path = ?",
                                (layout_spans_json, path_key),
                            )
                        connection.execute(
                            "DELETE FROM import_issues WHERE normalized_absolute_path = ?",
                            (path_key,),
                        )
                    unchanged += 1
                continue

            work_id = _stable_id("work", path_key)
            document_id = _stable_id("document", path_key)
            now = datetime.now(timezone.utc).isoformat()
            try:
                with connection:
                    connection.execute(
                        "INSERT INTO works (work_id, provisional_title, created_at) "
                        "VALUES (?, ?, ?)",
                        (work_id, source_file.path.stem, now),
                    )
                    connection.execute(
                        """
                        INSERT INTO documents (
                            document_id, work_id, source_class, source_path,
                            normalized_absolute_path, filename, raw_sha256,
                            normalized_text, recovery_warnings, import_state, imported_at,
                            layout_spans_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?)
                        """,
                        (
                            document_id,
                            work_id,
                            source_file.source_class,
                            str(source_file.path),
                            path_key,
                            source_file.path.name,
                            digest,
                            normalized_text,
                            json.dumps(list(recovery_warnings), ensure_ascii=False),
                            now,
                            layout_spans_json,
                        ),
                    )
                    connection.execute(
                        "DELETE FROM import_issues WHERE normalized_absolute_path = ?",
                        (path_key,),
                    )
            except sqlite3.IntegrityError as exc:
                raise PersistenceError(
                    "Could not save {!s}: {}".format(source_file.path, exc)
                ) from exc
            added += 1

        for unsupported in scan.unsupported:
            path_key = _normalized_absolute_path(unsupported.path)
            _record_issue(
                connection,
                path_key,
                str(unsupported.path),
                unsupported.source_class,
                "unsupported",
                "Unsupported file type: .{}".format(unsupported.file_format),
            )

    return ImportResult(
        added=added,
        unchanged=unchanged,
        conflicts=conflicts,
        failed=failed,
        unsupported=len(scan.unsupported),
    )


def get_status(config: ProjectConfig) -> Dict[str, object]:
    """Return saved work/document counts and unresolved per-file issues."""
    with _connection(config, _configured_roots(config)) as connection:
        counts = {
            "works": connection.execute("SELECT COUNT(*) FROM works").fetchone()[0],
            "documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        }
        vector_index = _vector_status(connection)
        issues: List[Dict[str, str]] = [
            dict(row)
            for row in connection.execute(
                "SELECT source_path, source_class, status, message "
                "FROM import_issues ORDER BY source_path"
            )
        ]
    return {
        "database": str(config.storage_directory / DATABASE_FILENAME),
        "works": counts["works"],
        "documents": counts["documents"],
        "issues": issues,
        "vector_index": vector_index,
    }


def _vector_status(connection: sqlite3.Connection) -> Dict[str, object]:
    expected_chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    indexed = connection.execute(
        "SELECT index_version, embedding_model, collection_name, status "
        "FROM vector_indexes WHERE status IN ('active', 'building') "
        "ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, created_at DESC LIMIT 1"
    ).fetchone()
    documents_with_chunks = connection.execute(
        "SELECT COUNT(*) FROM document_chunk_counts WHERE chunk_count > 0"
    ).fetchone()[0]
    if indexed is None:
        return {
            "status": "not_built",
            "index_version": None,
            "embedding_model": None,
            "collection_name": None,
            "expected_chunks": expected_chunks,
            "ready_chunks": 0,
            "failed_chunks": 0,
            "pending_chunks": expected_chunks,
            "documents_ready": 0,
            "documents_with_chunks": documents_with_chunks,
        }

    version = indexed["index_version"]
    job_counts = {
        row["status"]: row["count"]
        for row in connection.execute(
            "SELECT status, COUNT(*) AS count FROM embedding_jobs "
            "WHERE index_version = ? GROUP BY status",
            (version,),
        )
    }
    ready_chunks = job_counts.get("ready", 0)
    failed_chunks = job_counts.get("failed", 0)
    pending_chunks = max(0, expected_chunks - ready_chunks - failed_chunks)
    documents_ready = connection.execute(
        """
        SELECT COUNT(*) FROM documents AS d
        WHERE {}
        """.format(_READY_DOCUMENT_SQL),
        (version,),
    ).fetchone()[0]
    return {
        "status": indexed["status"],
        "index_version": version,
        "embedding_model": indexed["embedding_model"],
        "collection_name": indexed["collection_name"],
        "expected_chunks": expected_chunks,
        "ready_chunks": ready_chunks,
        "failed_chunks": failed_chunks,
        "pending_chunks": pending_chunks,
        "documents_ready": documents_ready,
        "documents_with_chunks": documents_with_chunks,
    }


def search_chunks(
    config: ProjectConfig,
    query: str,
    source_class: str = "canonical",
    work_id: Optional[str] = None,
    limit: int = 10,
    ready_index_version: Optional[str] = None,
    require_all_terms: bool = True,
    timings: Optional[SearchTimings] = None,
) -> List[Dict[str, object]]:
    """Search committed chunks with FTS5 and return exact source spans."""
    if source_class not in ("canonical", "fanfiction", "both"):
        raise ValueError("source_class must be canonical, fanfiction, or both")
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    if not terms:
        return []
    operator = " AND " if require_all_terms else " OR "
    fts_query = operator.join(
        '"{}"'.format(term.replace('"', '""')) for term in terms
    )

    conditions = ["chunk_fts MATCH ?"]
    parameters: List[object] = [fts_query]
    if source_class == "both":
        conditions.append("d.source_class IN (?, ?)")
        parameters.extend(("canonical", "fanfiction"))
    else:
        conditions.append("d.source_class = ?")
        parameters.append(source_class)
    if work_id is not None:
        conditions.append("d.work_id = ?")
        parameters.append(work_id)
    if ready_index_version is not None:
        conditions.append(_READY_DOCUMENT_SQL)
        parameters.append(ready_index_version)
    parameters.append(limit)

    sql = """
        SELECT c.chunk_id, c.document_id, d.work_id, d.filename, d.source_path,
               d.source_class, c.chunk_type, c.chunk_text, c.chunk_ordinal,
               bm25(chunk_fts) AS rank
        FROM chunk_fts
        JOIN chunks AS c ON c.rowid = chunk_fts.rowid
        JOIN documents AS d ON d.document_id = c.document_id
        WHERE {}
        ORDER BY rank, d.source_path, c.chunk_ordinal
        LIMIT ?
    """.format(" AND ".join(conditions))

    results: List[Dict[str, object]] = []
    with _connection(config, _configured_roots(config)) as connection:
        with measure(timings, "fts.sql"):
            rows = connection.execute(sql, parameters).fetchall()
        with measure(timings, "fts.citations"):
            results.extend(_format_chunk_rows(connection, rows))
    if timings is not None:
        timings.increment("fts.candidates", len(results))
    return results


def _format_chunk_rows(connection, rows) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for row in rows:
        spans = [
            dict(span)
            for span in connection.execute(
                "SELECT span_ordinal, start_offset, end_offset "
                "FROM chunk_spans WHERE chunk_id = ? ORDER BY span_ordinal",
                (row["chunk_id"],),
            )
        ]
        speakers: List[str] = []
        scene_headings: List[str] = []
        turn_rows = connection.execute(
            """
            SELECT t.turn_ordinal, t.original_speaker_label, s.scene_heading
            FROM chunk_spans AS cs
            JOIN dialogue_turns AS t
              ON t.document_id = ?
             AND t.start_offset < cs.end_offset
             AND t.end_offset > cs.start_offset
            LEFT JOIN scenes AS s ON s.scene_id = t.scene_id
            WHERE cs.chunk_id = ?
            ORDER BY t.turn_ordinal
            """,
            (row["document_id"], row["chunk_id"]),
        ).fetchall()
        for turn in turn_rows:
            label = turn["original_speaker_label"]
            heading = turn["scene_heading"]
            if label and label not in speakers:
                speakers.append(label)
            if heading and heading not in scene_headings:
                scene_headings.append(heading)
        results.append(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "work_id": row["work_id"],
                "filename": row["filename"],
                "source_path": row["source_path"],
                "source_class": row["source_class"],
                "chunk_type": row["chunk_type"],
                "chunk_text": row["chunk_text"],
                "chunk_ordinal": row["chunk_ordinal"],
                "rank": row["rank"],
                "spans": spans,
                "speaker_labels": speakers,
                "scene_headings": scene_headings,
            }
        )
    return results


def search_chunks_by_ids(
    config: ProjectConfig,
    chunk_ids: Sequence[str],
    source_class: str,
    work_id: Optional[str],
    index_version: str,
    limit: Optional[int] = None,
    timings: Optional[SearchTimings] = None,
) -> List[Dict[str, object]]:
    """Resolve vector candidates through SQLite and keep only ready documents."""
    if source_class not in ("canonical", "fanfiction", "both"):
        raise ValueError("source_class must be canonical, fanfiction, or both")
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero")
    ordered_ids = list(dict.fromkeys(chunk_ids))
    if not ordered_ids:
        return []

    conditions = [
        "c.chunk_id IN ({})".format(",".join("?" for _ in ordered_ids)),
        "EXISTS (SELECT 1 FROM embedding_jobs AS j "
        "WHERE j.chunk_id = c.chunk_id AND j.index_version = ? "
        "AND j.status = 'ready')",
        _READY_DOCUMENT_SQL,
    ]
    parameters: List[object] = list(ordered_ids)
    parameters.extend((index_version, index_version))
    if source_class == "both":
        conditions.append("d.source_class IN (?, ?)")
        parameters.extend(("canonical", "fanfiction"))
    else:
        conditions.append("d.source_class = ?")
        parameters.append(source_class)
    if work_id is not None:
        conditions.append("d.work_id = ?")
        parameters.append(work_id)

    sql = """
        SELECT c.chunk_id, c.document_id, d.work_id, d.filename, d.source_path,
               d.source_class, c.chunk_type, c.chunk_text, c.chunk_ordinal,
               NULL AS rank
        FROM chunks AS c
        JOIN documents AS d ON d.document_id = c.document_id
        WHERE {}
    """.format(" AND ".join(conditions))

    with _connection(config, _configured_roots(config)) as connection:
        with measure(timings, "vector.sqlite_lookup"):
            rows = connection.execute(sql, parameters).fetchall()
            rows_by_id = {row["chunk_id"]: row for row in rows}
            ordered_rows = [rows_by_id[chunk_id] for chunk_id in ordered_ids if chunk_id in rows_by_id]
            if limit is not None:
                ordered_rows = ordered_rows[:limit]
        with measure(timings, "vector.citations"):
            results = _format_chunk_rows(connection, ordered_rows)
        if timings is not None:
            timings.increment("vector.validated_candidates", len(results))
        return results


def validate_database(config: ProjectConfig) -> Dict[str, object]:
    """Check FTS consistency and ensure every chunk reconstructs from its spans."""
    errors: List[str] = []
    with _connection(config, _configured_roots(config)) as connection:
        connection.execute(
            "INSERT INTO chunk_fts(chunk_fts, rank) VALUES ('integrity-check', 1)"
        )
        document_count = connection.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        indexed_count = connection.execute(
            "SELECT COUNT(*) FROM chunk_fts"
        ).fetchone()[0]
        if indexed_count != chunk_count:
            errors.append(
                "FTS content rows do not match chunks ({} vs {}).".format(
                    indexed_count, chunk_count
                )
            )

        documents = connection.execute(
            "SELECT document_id, normalized_text FROM documents ORDER BY document_id"
        )
        for document in documents:
            chunks = connection.execute(
                """
                SELECT c.chunk_id, c.chunk_text, cs.span_ordinal,
                       cs.start_offset, cs.end_offset
                FROM chunks AS c
                LEFT JOIN chunk_spans AS cs ON cs.chunk_id = c.chunk_id
                WHERE c.document_id = ?
                ORDER BY c.chunk_ordinal, cs.span_ordinal
                """,
                (document["document_id"],),
            )
            current_chunk_id = None
            current_chunk_text = ""
            source_spans: List[str] = []

            def check_current_chunk() -> None:
                if current_chunk_id is None:
                    return
                if not source_spans:
                    errors.append("Chunk {} has no source spans.".format(current_chunk_id))
                elif COMPOSITE_SEPARATOR.join(source_spans) != current_chunk_text:
                    errors.append(
                        "Chunk {} does not reconstruct from its source spans.".format(
                            current_chunk_id
                        )
                    )

            for row in chunks:
                if row["chunk_id"] != current_chunk_id:
                    check_current_chunk()
                    current_chunk_id = row["chunk_id"]
                    current_chunk_text = row["chunk_text"]
                    source_spans = []
                if row["span_ordinal"] is not None:
                    start, end = row["start_offset"], row["end_offset"]
                    text = document["normalized_text"]
                    if not (0 <= start < end <= len(text)):
                        errors.append(
                            "Chunk {} has an out-of-range span [{}, {}).".format(
                                current_chunk_id, start, end
                            )
                        )
                    else:
                        source_spans.append(text[start:end])
            check_current_chunk()

    return {
        "documents": document_count,
        "chunks": chunk_count,
        "fts_rows": indexed_count,
        "errors": errors,
        "valid": not errors,
    }


def _persist_parse_result(
    connection: sqlite3.Connection, document: sqlite3.Row, result: ParseResult
) -> None:
    normalized_text = document["normalized_text"]
    validate_parse_result(document["document_id"], normalized_text, result)
    document_id = document["document_id"]
    with connection:
        connection.execute("DELETE FROM dialogue_turns WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM parse_warnings WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM scenes WHERE document_id = ?", (document_id,))

        for scene in result.scenes:
            connection.execute(
                """
                INSERT INTO scenes (
                    scene_id, document_id, start_offset, end_offset, scene_text,
                    scene_heading, scene_ordinal, scene_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene.scene_id,
                    document_id,
                    scene.start_offset,
                    scene.end_offset,
                    scene.scene_text,
                    scene.scene_heading,
                    scene.scene_ordinal,
                    scene.scene_type,
                ),
            )
        for turn in result.dialogue_turns:
            connection.execute(
                """
                INSERT INTO dialogue_turns (
                    turn_id, document_id, start_offset, end_offset, turn_text,
                    scene_id, original_speaker_label, speaker_character_id,
                    attribution_status, turn_ordinal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.turn_id,
                    document_id,
                    turn.start_offset,
                    turn.end_offset,
                    turn.turn_text,
                    turn.scene_id,
                    turn.original_speaker_label,
                    turn.speaker_character_id,
                    turn.attribution_status,
                    turn.turn_ordinal,
                ),
            )
        for chunk in result.chunks:
            first_span = chunk.spans[0]
            last_span = chunk.spans[-1]
            connection.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_type, start_offset, end_offset,
                    chunk_text, chunker_version, chunk_ordinal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    document_id,
                    chunk.chunk_type,
                    first_span.start_offset,
                    last_span.end_offset,
                    chunk.chunk_text,
                    chunk.chunker_version,
                    chunk.chunk_ordinal,
                ),
            )
            for span_ordinal, span in enumerate(chunk.spans):
                connection.execute(
                    """
                    INSERT INTO chunk_spans (
                        chunk_id, span_ordinal, start_offset, end_offset
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        span_ordinal,
                        span.start_offset,
                        span.end_offset,
                    ),
                )
        for warning in result.warnings:
            connection.execute(
                """
                INSERT INTO parse_warnings (
                    warning_id, document_id, scene_id, start_offset, end_offset,
                    warning_code, message, parser_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    warning.warning_id,
                    document_id,
                    warning.scene_id,
                    warning.start_offset,
                    warning.end_offset,
                    warning.warning_code,
                    warning.message,
                    warning.parser_version,
                ),
            )


def parse_documents(
    config: ProjectConfig,
    document_id: Optional[str] = None,
    parser_version: Optional[str] = None,
    chunker_version: Optional[str] = None,
) -> ParseSummary:
    """Parse all stored documents, or one selected document, transactionally."""
    with _connection(config, _configured_roots(config)) as connection:
        if document_id is None:
            documents = connection.execute(
                "SELECT document_id, source_class, normalized_text, layout_spans_json "
                "FROM documents ORDER BY document_id"
            ).fetchall()
        else:
            document = connection.execute(
                "SELECT document_id, source_class, normalized_text, layout_spans_json "
                "FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if document is None:
                raise PersistenceError("Document not found: {}".format(document_id))
            documents = [document]

        parsed_count = scene_count = turn_count = chunk_count = warning_count = 0
        errors: List[str] = []
        for document in documents:
            try:
                options = {}
                options["layout_spans"] = json.loads(
                    document["layout_spans_json"] or "[]"
                )
                if parser_version is not None:
                    options["parser_version"] = parser_version
                if chunker_version is not None:
                    options["chunker_version"] = chunker_version
                result = parse_document(
                    document["document_id"],
                    document["source_class"],
                    document["normalized_text"],
                    **options
                )
                _persist_parse_result(connection, document, result)
            except (ValueError, sqlite3.Error) as exc:
                errors.append("{}: {}".format(document["document_id"], exc))
                continue
            parsed_count += 1
            scene_count += len(result.scenes)
            turn_count += len(result.dialogue_turns)
            chunk_count += len(result.chunks)
            warning_count += len(result.warnings)

    return ParseSummary(
        documents_parsed=parsed_count,
        scenes=scene_count,
        dialogue_turns=turn_count,
        chunks=chunk_count,
        warnings=warning_count,
        errors=tuple(errors),
    )


def inspect_document(config: ProjectConfig, document_id: str) -> Dict[str, object]:
    """Return parsed records hydrated with exact source snippets for review."""
    with _connection(config, _configured_roots(config)) as connection:
        document = connection.execute(
            "SELECT document_id, filename, source_class, normalized_text "
            "FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if document is None:
            raise PersistenceError("Document not found: {}".format(document_id))
        text = document["normalized_text"]
        scenes = []
        for row in connection.execute(
            """
            SELECT scene_id, scene_ordinal, scene_type, scene_heading,
                   start_offset, end_offset
            FROM scenes WHERE document_id = ? ORDER BY scene_ordinal
            """,
            (document_id,),
        ):
            item = dict(row)
            item["source_snippet"] = text[item["start_offset"] : item["end_offset"]]
            scenes.append(item)

        turns = []
        for row in connection.execute(
            """
            SELECT turn_id, scene_id, turn_ordinal, start_offset, end_offset,
                   original_speaker_label, speaker_character_id,
                   attribution_status
            FROM dialogue_turns WHERE document_id = ? ORDER BY turn_ordinal
            """,
            (document_id,),
        ):
            item = dict(row)
            item["source_snippet"] = text[item["start_offset"] : item["end_offset"]]
            turns.append(item)

        chunks = []
        for row in connection.execute(
            """
            SELECT chunk_id, chunk_type, chunker_version, chunk_ordinal,
                   start_offset, end_offset, chunk_text
            FROM chunks WHERE document_id = ?
            ORDER BY chunk_ordinal
            """,
            (document_id,),
        ):
            item = dict(row)
            spans = []
            for span_row in connection.execute(
                """
                SELECT span_ordinal, start_offset, end_offset
                FROM chunk_spans WHERE chunk_id = ? ORDER BY span_ordinal
                """,
                (item["chunk_id"],),
            ):
                span = dict(span_row)
                span["source_snippet"] = text[
                    span["start_offset"] : span["end_offset"]
                ]
                spans.append(span)
            item["spans"] = spans
            chunks.append(item)

        warnings = []
        for row in connection.execute(
            """
            SELECT warning_id, scene_id, start_offset, end_offset, warning_code,
                   message, parser_version
            FROM parse_warnings WHERE document_id = ? ORDER BY start_offset, warning_id
            """,
            (document_id,),
        ):
            item = dict(row)
            item["source_snippet"] = text[item["start_offset"] : item["end_offset"]]
            warnings.append(item)

    return {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "source_class": document["source_class"],
        "scenes": scenes,
        "dialogue_turns": turns,
        "chunks": chunks,
        "warnings": warnings,
    }