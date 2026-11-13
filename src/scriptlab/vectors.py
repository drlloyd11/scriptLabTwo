"""Resumable local embedding and Chroma indexing for ScriptLab chunks."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

from scriptlab.config import ProjectConfig
from scriptlab.corpus import _assert_path_has_no_reparse_components
from scriptlab.database import (
    PersistenceError,
    _READY_DOCUMENT_SQL,
    _configured_roots,
    _connection,
    get_status,
    search_chunks_by_ids,
)
from scriptlab.timing import SearchTimings, measure

_MAX_CONSECUTIVE_BATCH_FAILURES = 3


class VectorIndexError(RuntimeError):
    """Raised when local embedding or vector indexing cannot proceed safely."""


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""


class VectorCollection(Protocol):
    def existing_ids(self, chunk_ids: Sequence[str]) -> Set[str]: ...

    def upsert(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, object]],
    ) -> None: ...

    def count(self) -> int: ...

    def all_ids(self) -> List[str]: ...

    def delete(self, chunk_ids: Sequence[str]) -> None: ...

    def query(
        self,
        query_embedding: Sequence[float],
        limit: int,
        where: Optional[Dict[str, object]] = None,
    ) -> List[Tuple[str, float]]: ...


class OllamaEmbeddingProvider:
    """Minimal adapter for Ollama's local `/api/embed` endpoint."""

    def __init__(self, model: str, ollama_url: str, timeout: float = 120.0):
        self.model = model
        self.endpoint = ollama_url.rstrip("/") + "/api/embed"
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode(
            "utf-8"
        )
        result = None
        for attempt in range(3):
            split_after_retry = False
            request = Request(
                self.endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:500]
                transient = exc.code in (429, 500, 502, 503, 504) or (
                    exc.code == 400
                    and any(
                        marker in body.casefold()
                        for marker in ("/tokenize", "connectex", "connection refused")
                    )
                )
                if transient and attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                split_after_retry = (
                    exc.code == 400
                    and any(
                        marker in body.casefold()
                        for marker in ("/tokenize", "connectex", "connection refused")
                    )
                    and len(texts) > 1
                )
                if split_after_retry:
                    midpoint = len(texts) // 2
                    return self.embed(texts[:midpoint]) + self.embed(texts[midpoint:])
                raise VectorIndexError(
                    "Ollama embedding request failed for model {!r}: HTTP {} {}".format(
                        self.model, exc.code, body or exc.reason
                    )
                ) from exc
            except (URLError, OSError) as exc:
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise VectorIndexError(
                    "Ollama embedding request failed for model {!r}: {}".format(
                        self.model, exc
                    )
                ) from exc
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise VectorIndexError(
                    "Ollama returned an invalid embedding response: {}".format(exc)
                ) from exc

        if not isinstance(result, dict):
            raise VectorIndexError("Ollama returned an invalid embedding response.")
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise VectorIndexError(
                "Ollama returned an unexpected number of embedding vectors."
            )
        return _validate_embeddings(embeddings, len(texts))


class ChromaVectorCollection:
    """Persistent Chroma collection adapter stored under ScriptLab storage."""

    def __init__(
        self,
        config: ProjectConfig,
        collection_name: str,
        embedding_model: str,
        index_version: str,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise VectorIndexError(
                "ChromaDB is not installed. Install ScriptLab with the 'vectors' extra."
            ) from exc

        path = config.storage_directory / "chroma"
        try:
            path.mkdir(parents=True, exist_ok=True)
            path_info = _assert_path_has_no_reparse_components(
                path, allow_missing_leaf=False
            )
            if path_info is None or not path.is_dir():
                raise OSError("Chroma storage path is not a directory")
            self.client = chromadb.PersistentClient(path=str(path))
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": embedding_model,
                    "index_version": index_version,
                },
                embedding_function=None,
            )
        except Exception as exc:
            if isinstance(exc, VectorIndexError):
                raise
            raise VectorIndexError(
                "Could not open Chroma collection {!r}: {}".format(
                    collection_name, exc
                )
            ) from exc

        metadata = self.collection.metadata or {}
        if (
            metadata.get("embedding_model") != embedding_model
            or metadata.get("index_version") != index_version
        ):
            raise VectorIndexError(
                "Chroma collection metadata does not match the requested model/index version."
            )

    def existing_ids(self, chunk_ids: Sequence[str]) -> Set[str]:
        if not chunk_ids:
            return set()
        return set(self.collection.get(ids=list(chunk_ids), include=[])["ids"])

    def upsert(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, object]],
    ) -> None:
        self.collection.upsert(
            ids=list(chunk_ids),
            documents=list(texts),
            embeddings=[list(vector) for vector in embeddings],
            metadatas=list(metadatas),
        )

    def count(self) -> int:
        return self.collection.count()

    def all_ids(self) -> List[str]:
        return list(self.collection.get(include=[])["ids"])

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            self.collection.delete(ids=list(chunk_ids))

    def query(
        self,
        query_embedding: Sequence[float],
        limit: int,
        where: Optional[Dict[str, object]] = None,
    ) -> List[Tuple[str, float]]:
        options = {"query_embeddings": [list(query_embedding)], "n_results": limit}
        if where:
            options["where"] = where
        try:
            result = self.collection.query(include=["distances"], **options)
            ids = result.get("ids") or [[]]
            distances = result.get("distances") or [[]]
        except Exception as exc:
            raise VectorIndexError("Chroma vector search failed: {}".format(exc)) from exc
        if len(ids) != 1 or len(distances) != 1 or len(ids[0]) != len(distances[0]):
            raise VectorIndexError("Chroma returned malformed vector search results.")
        return [(chunk_id, float(distance)) for chunk_id, distance in zip(ids[0], distances[0])]


@dataclass(frozen=True)
class IndexSummary:
    index_version: str
    embedding_model: str
    chunks_expected: int
    chunks_indexed_this_run: int
    chunks_reused: int
    chunks_failed: int
    chunks_ready: int
    documents_ready: int
    index_ready: bool
    errors: Tuple[str, ...]


def _validate_embeddings(
    embeddings: Sequence[Sequence[object]], expected_count: int
) -> List[List[float]]:
    if len(embeddings) != expected_count:
        raise VectorIndexError(
            "Embedding provider returned {} vectors for {} chunks.".format(
                len(embeddings), expected_count
            )
        )
    result: List[List[float]] = []
    dimensions = None
    for vector in embeddings:
        try:
            normalized = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise VectorIndexError("Embedding provider returned a non-numeric vector.") from exc
        if not normalized or any(not math.isfinite(value) for value in normalized):
            raise VectorIndexError("Embedding vectors must be non-empty and finite.")
        if dimensions is None:
            dimensions = len(normalized)
        elif len(normalized) != dimensions:
            raise VectorIndexError("Embedding vectors have inconsistent dimensions.")
        result.append(normalized)
    return result


def _new_index_record(connection, embedding_model: str) -> Dict[str, str]:
    index_version = "v1-{}".format(uuid.uuid4().hex[:12])
    collection_name = "scriptlab_{}".format(index_version.replace("-", "_"))
    now = datetime.now(timezone.utc).isoformat()
    connection.execute(
        """
        INSERT INTO vector_indexes (
            index_version, embedding_model, collection_name, status,
            created_at, updated_at
        ) VALUES (?, ?, ?, 'building', ?, ?)
        """,
        (index_version, embedding_model, collection_name, now, now),
    )
    return {
        "index_version": index_version,
        "embedding_model": embedding_model,
        "collection_name": collection_name,
        "status": "building",
    }


def _select_index(
    connection, embedding_model: str, rebuild: bool
) -> Dict[str, str]:
    active = connection.execute(
        "SELECT index_version, embedding_model, collection_name, status "
        "FROM vector_indexes WHERE status = 'active'"
    ).fetchone()
    if active is not None and active["embedding_model"] != embedding_model and not rebuild:
        raise VectorIndexError(
            "Active vectors use model {!r}; pass --rebuild to index with {!r}.".format(
                active["embedding_model"], embedding_model
            )
        )
    if active is not None and active["embedding_model"] == embedding_model and not rebuild:
        return dict(active)

    building = connection.execute(
        "SELECT index_version, embedding_model, collection_name, status "
        "FROM vector_indexes WHERE status = 'building' AND embedding_model = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (embedding_model,),
    ).fetchone()
    if building is not None:
        return dict(building)
    with connection:
        return _new_index_record(connection, embedding_model)


def _record_jobs(
    connection,
    chunk_ids: Sequence[str],
    index_version: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    if not chunk_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    with connection:
        connection.executemany(
            """
            INSERT INTO embedding_jobs (
                chunk_id, index_version, status, error, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id, index_version) DO UPDATE SET
                status = excluded.status,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            [
                (chunk_id, index_version, status, error, now)
                for chunk_id in chunk_ids
            ],
        )


def _prune_stale_vectors(connection, collection: VectorCollection) -> None:
    all_ids = collection.all_ids()
    for start in range(0, len(all_ids), 500):
        vector_ids = all_ids[start : start + 500]
        placeholders = ",".join("?" for _ in vector_ids)
        expected = {
            row[0]
            for row in connection.execute(
                "SELECT chunk_id FROM chunks WHERE chunk_id IN ({})".format(
                    placeholders
                ),
                vector_ids,
            )
        }
        stale = [chunk_id for chunk_id in vector_ids if chunk_id not in expected]
        collection.delete(stale)


def _index_counts(connection, index_version: str) -> Tuple[int, int, int, int]:
    expected = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    ready = connection.execute(
        "SELECT COUNT(*) FROM embedding_jobs WHERE index_version = ? AND status = 'ready'",
        (index_version,),
    ).fetchone()[0]
    failed = connection.execute(
        "SELECT COUNT(*) FROM embedding_jobs WHERE index_version = ? AND status = 'failed'",
        (index_version,),
    ).fetchone()[0]
    documents_ready = connection.execute(
        """
        SELECT COUNT(*) FROM documents AS d
        WHERE {}
        """.format(_READY_DOCUMENT_SQL),
        (index_version,),
    ).fetchone()[0]
    return expected, ready, failed, documents_ready


def index_chunks(
    config: ProjectConfig,
    batch_size: int = 16,
    limit: Optional[int] = None,
    rebuild: bool = False,
    retry_failed_only: bool = False,
    embedding_provider: Optional[EmbeddingProvider] = None,
    collection_factory: Optional[Callable[..., VectorCollection]] = None,
    progress_callback: Optional[Callable[[int, int, int, int, int, int], None]] = None,
) -> IndexSummary:
    """Resume indexing SQLite chunks to the configured local vector model."""
    if batch_size < 1:
        raise ValueError("batch_size must be greater than zero")
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero")
    provider = embedding_provider or OllamaEmbeddingProvider(
        config.embedding_model, config.ollama_url
    )
    factory = collection_factory or ChromaVectorCollection

    with _connection(config, _configured_roots(config)) as connection:
        chunks_expected = connection.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        if not chunks_expected:
            raise VectorIndexError("No parsed chunks are available to index.")
        with connection:
            index = _select_index(connection, config.embedding_model, rebuild)
        if retry_failed_only:
            failed_count = connection.execute(
                "SELECT COUNT(*) FROM embedding_jobs "
                "WHERE index_version = ? AND status = 'failed'",
                (index["index_version"],),
            ).fetchone()[0]
            if not failed_count:
                raise VectorIndexError(
                    "No failed chunks are recorded for this vector index."
                )
            progress_total = failed_count
        else:
            progress_total = chunks_expected
        collection = factory(
            config,
            index["collection_name"],
            config.embedding_model,
            index["index_version"],
        )
        indexed_this_run = reused = failed_this_run = attempted = 0
        consecutive_failures = 0
        errors: List[str] = []
        last_chunk_id = ""
        completed_scan = True
        rows_scanned = 0
        job_counts = {
            row["status"]: row["count"]
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM embedding_jobs "
                "WHERE index_version = ? GROUP BY status",
                (index["index_version"],),
            )
        }
        ready_count = job_counts.get("ready", 0)
        failed_count = job_counts.get("failed", 0)

        while True:
            if retry_failed_only:
                rows = connection.execute(
                    """
                    SELECT c.chunk_id, c.document_id, c.chunk_text, d.work_id,
                           d.filename, d.source_class, c.chunk_type
                    FROM embedding_jobs AS j
                    JOIN chunks AS c ON c.chunk_id = j.chunk_id
                    JOIN documents AS d ON d.document_id = c.document_id
                    WHERE j.index_version = ? AND j.status = 'failed'
                      AND c.chunk_id > ?
                    ORDER BY c.chunk_id
                    LIMIT ?
                    """,
                    (index["index_version"], last_chunk_id, batch_size),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT c.chunk_id, c.document_id, c.chunk_text, d.work_id,
                           d.filename, d.source_class, c.chunk_type
                    FROM chunks AS c
                    JOIN documents AS d ON d.document_id = c.document_id
                    WHERE c.chunk_id > ?
                    ORDER BY c.chunk_id
                    LIMIT ?
                    """,
                    (last_chunk_id, batch_size),
                ).fetchall()
            if not rows:
                break
            rows_scanned += len(rows)
            last_chunk_id = rows[-1]["chunk_id"]
            chunk_ids = [row["chunk_id"] for row in rows]
            present_ids = collection.existing_ids(chunk_ids)
            job_status = {
                row["chunk_id"]: row["status"]
                for row in connection.execute(
                    "SELECT chunk_id, status FROM embedding_jobs "
                    "WHERE index_version = ? AND chunk_id IN ({})".format(
                        ",".join("?" for _ in chunk_ids)
                    ),
                    [index["index_version"]] + chunk_ids,
                )
            }

            verified_ids = [
                chunk_id
                for chunk_id in chunk_ids
                if chunk_id in present_ids and job_status.get(chunk_id) != "ready"
            ]
            for chunk_id in verified_ids:
                previous_status = job_status.get(chunk_id)
                ready_count += 1
                if previous_status == "failed":
                    failed_count -= 1
            _record_jobs(connection, verified_ids, index["index_version"], "ready")
            reused += sum(
                1
                for chunk_id in chunk_ids
                if chunk_id in present_ids and job_status.get(chunk_id) == "ready"
            )

            missing_rows = [row for row in rows if row["chunk_id"] not in present_ids]
            if limit is not None:
                remaining = limit - attempted
                if remaining <= 0:
                    completed_scan = False
                    break
                missing_rows = missing_rows[:remaining]
            if not missing_rows:
                if progress_callback is not None:
                    progress_callback(
                        rows_scanned,
                        progress_total,
                        ready_count,
                        indexed_this_run,
                        reused,
                        failed_count,
                    )
                continue

            batch_ids = [row["chunk_id"] for row in missing_rows]
            batch_texts = [row["chunk_text"] for row in missing_rows]
            attempted += len(batch_ids)
            try:
                embeddings = _validate_embeddings(
                    provider.embed(batch_texts), len(batch_ids)
                )
                metadata = [
                    {
                        "document_id": row["document_id"],
                        "work_id": row["work_id"],
                        "filename": row["filename"],
                        "source_class": row["source_class"],
                        "chunk_type": row["chunk_type"],
                        "index_version": index["index_version"],
                        "embedding_model": config.embedding_model,
                    }
                    for row in missing_rows
                ]
                collection.upsert(batch_ids, batch_texts, embeddings, metadata)
                verified_after_upsert = collection.existing_ids(batch_ids)
                ready_ids = [
                    chunk_id for chunk_id in batch_ids if chunk_id in verified_after_upsert
                ]
                missing_after_upsert = [
                    chunk_id for chunk_id in batch_ids if chunk_id not in verified_after_upsert
                ]
                _record_jobs(connection, ready_ids, index["index_version"], "ready")
                for chunk_id in ready_ids:
                    previous_status = job_status.get(chunk_id)
                    if previous_status != "ready":
                        ready_count += 1
                    if previous_status == "failed":
                        failed_count -= 1
                _record_jobs(
                    connection,
                    missing_after_upsert,
                    index["index_version"],
                    "failed",
                    "Vector store did not return an upserted chunk ID.",
                )
                for chunk_id in missing_after_upsert:
                    previous_status = job_status.get(chunk_id)
                    if previous_status == "ready":
                        ready_count -= 1
                    if previous_status != "failed":
                        failed_count += 1
                indexed_this_run += len(ready_ids)
                failed_this_run += len(missing_after_upsert)
                if missing_after_upsert:
                    consecutive_failures += 1
                    errors.append(
                        "Vector store did not verify {} upserted chunk(s).".format(
                            len(missing_after_upsert)
                        )
                    )
                else:
                    consecutive_failures = 0
            except Exception as exc:
                error = "{}: {}".format(type(exc).__name__, exc)
                _record_jobs(
                    connection, batch_ids, index["index_version"], "failed", error
                )
                for chunk_id in batch_ids:
                    previous_status = job_status.get(chunk_id)
                    if previous_status == "ready":
                        ready_count -= 1
                    if previous_status != "failed":
                        failed_count += 1
                failed_this_run += len(batch_ids)
                consecutive_failures += 1
                errors.append(error)

            if progress_callback is not None:
                progress_callback(
                    rows_scanned,
                    progress_total,
                    ready_count,
                    indexed_this_run,
                    reused,
                    failed_count,
                )

            if consecutive_failures >= _MAX_CONSECUTIVE_BATCH_FAILURES:
                completed_scan = False
                break

            if limit is not None and attempted >= limit:
                completed_scan = False
                break

        expected, ready, failed, documents_ready = _index_counts(
            connection, index["index_version"]
        )
        if progress_callback is not None:
            progress_callback(
                rows_scanned,
                progress_total,
                ready,
                indexed_this_run,
                reused,
                failed,
            )
        if completed_scan and ready == expected and not failed:
            if collection.count() > expected:
                _prune_stale_vectors(connection, collection)
            vector_count = collection.count()
            if vector_count != expected:
                errors.append(
                    "Vector collection has {} IDs; SQLite expects {}.".format(
                        vector_count, expected
                    )
                )
            elif expected:
                now = datetime.now(timezone.utc).isoformat()
                with connection:
                    connection.execute(
                        "UPDATE vector_indexes SET status = 'retired', updated_at = ? "
                        "WHERE status = 'active' AND index_version != ?",
                        (now, index["index_version"]),
                    )
                    connection.execute(
                        "UPDATE vector_indexes SET status = 'active', updated_at = ? "
                        "WHERE index_version = ?",
                        (now, index["index_version"]),
                    )
                index["status"] = "active"
        else:
            index["status"] = "building"

    return IndexSummary(
        index_version=index["index_version"],
        embedding_model=config.embedding_model,
        chunks_expected=expected,
        chunks_indexed_this_run=indexed_this_run,
        chunks_reused=reused,
        chunks_failed=failed_this_run,
        chunks_ready=ready,
        documents_ready=documents_ready,
        index_ready=index.get("status") == "active",
        errors=tuple(errors),
    )


def search_vector_chunks(
    config: ProjectConfig,
    query: str,
    source_class: str = "canonical",
    work_id: Optional[str] = None,
    limit: int = 10,
    embedding_provider: Optional[EmbeddingProvider] = None,
    collection_factory: Optional[Callable[..., VectorCollection]] = None,
    timings: Optional[SearchTimings] = None,
) -> Dict[str, object]:
    """Search Chroma candidates, then validate their provenance in SQLite."""
    if source_class not in ("canonical", "fanfiction", "both"):
        raise ValueError("source_class must be canonical, fanfiction, or both")
    if not query.strip():
        raise ValueError("query must not be empty")
    if limit < 1:
        raise ValueError("limit must be greater than zero")

    with measure(timings, "vector.status"):
        status = get_status(config)
    vector_status = status["vector_index"]
    documents_total = status["documents"]
    documents_ready = vector_status["documents_ready"]
    if vector_status["status"] != "active":
        raise VectorIndexError(
            "No complete active vector index is available ({} of {} imported documents ready).".format(
                documents_ready, documents_total
            )
        )
    if vector_status["embedding_model"] != config.embedding_model:
        raise VectorIndexError(
            "Active vectors use model {!r}, but the configured query model is {!r}.".format(
                vector_status["embedding_model"], config.embedding_model
            )
        )

    index_version = vector_status["index_version"]
    with measure(timings, "vector.collection_open"):
        collection = (collection_factory or ChromaVectorCollection)(
            config,
            vector_status["collection_name"],
            vector_status["embedding_model"],
            index_version,
        )
    with measure(timings, "vector.collection_count"):
        total_vectors = collection.count()
    if not total_vectors:
        raise VectorIndexError("The active vector collection is empty.")
    provider = embedding_provider or OllamaEmbeddingProvider(
        config.embedding_model, config.ollama_url
    )
    with measure(timings, "vector.embedding"):
        query_embedding = _validate_embeddings(provider.embed([query]), 1)[0]

    filters: List[Dict[str, object]] = []
    if source_class != "both":
        filters.append({"source_class": {"$eq": source_class}})
    if work_id is not None:
        filters.append({"work_id": {"$eq": work_id}})
    where = None
    if len(filters) == 1:
        where = filters[0]
    elif filters:
        where = {"$and": filters}

    candidate_limit = min(total_vectors, max(limit * 10, limit))
    resolved: List[Dict[str, object]] = []
    while True:
        if timings is not None:
            timings.increment("vector.query_calls")
            timings.increment("vector.requested_candidates", candidate_limit)
        with measure(timings, "vector.query"):
            candidates = collection.query(query_embedding, candidate_limit, where=where)
        if timings is not None:
            timings.increment("vector.returned_candidates", len(candidates))
        candidate_ids = [chunk_id for chunk_id, _ in candidates]
        with measure(timings, "vector.validation"):
            validated = search_chunks_by_ids(
                config,
                candidate_ids,
                source_class,
                work_id,
                index_version,
                limit=limit,
                timings=timings,
            )
        validated_by_id = {item["chunk_id"]: item for item in validated}
        resolved = []
        for rank, (chunk_id, distance) in enumerate(candidates, start=1):
            item = validated_by_id.get(chunk_id)
            if item is not None:
                item["rank"] = rank
                item["distance"] = distance
                resolved.append(item)
        if len(resolved) >= limit or candidate_limit >= total_vectors:
            break
        candidate_limit = min(total_vectors, candidate_limit * 2)

    return {
        "results": resolved[:limit],
        "index_version": index_version,
        "coverage": {
            "documents_ready": documents_ready,
            "documents_total": documents_total,
        },
    }