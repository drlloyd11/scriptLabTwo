from dataclasses import replace
from io import BytesIO
import json
import sqlite3
from urllib.error import HTTPError

import pytest

from scriptlab.config import load_config
from scriptlab.database import import_sources, parse_documents, get_status, search_chunks
import scriptlab.vectors as vector_module
from scriptlab.vectors import (
    OllamaEmbeddingProvider,
    VectorIndexError,
    index_chunks,
    search_vector_chunks,
)


def make_config(tmp_path):
    scripts = tmp_path / "scripts"
    fanfiction = tmp_path / "fanfiction"
    scripts.mkdir()
    fanfiction.mkdir()
    config_path = tmp_path / "scriptlab.yaml"
    config_path.write_text(
        "project_name: test\n"
        "source_directories:\n"
        "  canonical: ./scripts\n"
        "  fanfiction: ./fanfiction\n"
        "storage_directory: ./data\n"
        "embedding_model: test-embedding\n"
        "ollama_url: http://127.0.0.1:11434\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    source = scripts / "episode.txt"
    source.write_text(
        "INT. LIBRARY - NIGHT\n\n"
        "BUFFY\nThe hidden book holds the secret.\n\n"
        "GILES\nWe should read it carefully.\n\n"
        "EXT. ROAD - NIGHT\n\n"
        "XANDER\nI will keep watch.\n",
        encoding="utf-8",
    )
    import_sources(config)
    summary = parse_documents(config)
    assert not summary.errors
    return config


class FakeEmbeddingProvider:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [
            [float(len(text)), float(sum(ord(character) for character in text) % 997)]
            for text in texts
        ]


class FailOnceEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self):
        super().__init__()
        self.should_fail = True

    def embed(self, texts):
        if self.should_fail:
            self.should_fail = False
            self.calls += 1
            raise RuntimeError("synthetic provider interruption")
        return super().embed(texts)


class AlwaysFailingEmbeddingProvider(FakeEmbeddingProvider):
    def embed(self, texts):
        self.calls += 1
        raise RuntimeError("synthetic persistent outage")


class FakeVectorCollection:
    def __init__(self):
        self.records = {}
        self.queries = []

    def existing_ids(self, chunk_ids):
        return set(chunk_id for chunk_id in chunk_ids if chunk_id in self.records)

    def upsert(self, chunk_ids, texts, embeddings, metadatas):
        for chunk_id, text, embedding, metadata in zip(
            chunk_ids, texts, embeddings, metadatas
        ):
            self.records[chunk_id] = (text, list(embedding), dict(metadata))

    def count(self):
        return len(self.records)

    def all_ids(self):
        return list(self.records)

    def delete(self, chunk_ids):
        for chunk_id in chunk_ids:
            self.records.pop(chunk_id, None)

    def query(self, query_embedding, limit, where=None):
        self.queries.append((list(query_embedding), limit, where))
        return [
            (chunk_id, float(rank))
            for rank, chunk_id in enumerate(sorted(self.records), start=1)
        ][:limit]


class FakeCollectionFactory:
    def __init__(self):
        self.collections = {}

    def __call__(self, config, collection_name, embedding_model, index_version):
        return self.collections.setdefault(collection_name, FakeVectorCollection())


def _chunk_count(config):
    with sqlite3.connect(config.storage_directory / "scriptlab.sqlite3") as connection:
        return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def _index_rows(config):
    with sqlite3.connect(config.storage_directory / "scriptlab.sqlite3") as connection:
        return connection.execute(
            "SELECT index_version, embedding_model, collection_name, status "
            "FROM vector_indexes ORDER BY created_at"
        ).fetchall()


def test_index_resumes_without_duplicate_vectors_and_tracks_readiness(tmp_path):
    config = make_config(tmp_path)
    total = _chunk_count(config)
    assert total > 2
    provider = FakeEmbeddingProvider()
    collections = FakeCollectionFactory()

    first = index_chunks(
        config,
        batch_size=2,
        limit=1,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert first.chunks_indexed_this_run == 1
    assert not first.index_ready
    first_version = first.index_version
    status = get_status(config)["vector_index"]
    assert status["status"] == "building"
    assert status["ready_chunks"] == 1
    assert status["pending_chunks"] == total - 1
    assert status["documents_ready"] == 0

    second = index_chunks(
        config,
        batch_size=2,
        limit=1,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert second.index_version == first_version
    assert second.chunks_indexed_this_run == 1
    collection_name = _index_rows(config)[0][2]
    assert collections.collections[collection_name].count() == 2

    completed = index_chunks(
        config,
        batch_size=2,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert completed.index_ready
    assert completed.chunks_ready == total
    assert completed.documents_ready == 1
    assert collections.collections[collection_name].count() == total
    calls_after_completion = provider.calls

    repeated = index_chunks(
        config,
        batch_size=2,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert repeated.index_ready
    assert repeated.chunks_indexed_this_run == 0
    assert repeated.chunks_reused == total
    assert provider.calls == calls_after_completion


def test_index_reports_progress_after_batches(tmp_path):
    config = make_config(tmp_path)
    total = _chunk_count(config)
    updates = []

    result = index_chunks(
        config,
        batch_size=1,
        embedding_provider=FakeEmbeddingProvider(),
        collection_factory=FakeCollectionFactory(),
        progress_callback=lambda *values: updates.append(values),
    )

    assert updates
    assert updates[-1] == (
        total,
        total,
        total,
        total,
        0,
        0,
    )
    assert [update[2] for update in updates] == sorted(
        update[2] for update in updates
    )
    assert result.index_ready


def test_stale_vectors_are_pruned_before_index_activation(tmp_path):
    config = make_config(tmp_path)
    total = _chunk_count(config)

    class StaleCollectionFactory(FakeCollectionFactory):
        def __call__(self, config, collection_name, embedding_model, index_version):
            collection = super().__call__(
                config, collection_name, embedding_model, index_version
            )
            collection.upsert(["stale-vector"], ["stale"], [[1.0]], [{}])
            return collection

    collections = StaleCollectionFactory()
    result = index_chunks(
        config,
        embedding_provider=FakeEmbeddingProvider(),
        collection_factory=collections,
    )

    collection_name = _index_rows(config)[0][2]
    collection = collections.collections[collection_name]
    assert result.index_ready
    assert collection.count() == total
    assert "stale-vector" not in collection.all_ids()


def test_model_mismatch_requires_rebuild_and_failed_rebuild_keeps_active_index(tmp_path):
    config = make_config(tmp_path)
    collections = FakeCollectionFactory()
    initial = index_chunks(
        config,
        batch_size=2,
        embedding_provider=FakeEmbeddingProvider(),
        collection_factory=collections,
    )
    assert initial.index_ready
    changed_config = replace(config, embedding_model="replacement-model")

    with pytest.raises(VectorIndexError, match="pass --rebuild"):
        index_chunks(
            changed_config,
            embedding_provider=FakeEmbeddingProvider(),
            collection_factory=collections,
        )

    failed_rebuild = index_chunks(
        changed_config,
        batch_size=2,
        rebuild=True,
        embedding_provider=FailOnceEmbeddingProvider(),
        collection_factory=collections,
    )
    assert not failed_rebuild.index_ready
    rows = _index_rows(config)
    assert sum(row[3] == "active" for row in rows) == 1
    assert sum(row[3] == "building" for row in rows) == 1
    assert next(row for row in rows if row[3] == "active")[0] == initial.index_version

    resumed_rebuild = index_chunks(
        changed_config,
        batch_size=2,
        rebuild=True,
        embedding_provider=FakeEmbeddingProvider(),
        collection_factory=collections,
    )
    assert resumed_rebuild.index_ready
    assert resumed_rebuild.embedding_model == "replacement-model"
    rows = _index_rows(config)
    assert sum(row[3] == "active" for row in rows) == 1
    assert next(row for row in rows if row[3] == "active")[0] == resumed_rebuild.index_version
    assert sum(row[3] == "retired" for row in rows) == 1


def test_semantic_search_uses_sqlite_source_work_and_readiness_filters(tmp_path):
    config = make_config(tmp_path)
    fanfiction = config.fanfiction_directory / "story.txt"
    fanfiction.write_text(
        'Chapter One\n\n"The hidden book holds a secret," Willow said.\n',
        encoding="utf-8",
    )
    import_sources(config)
    parsed = parse_documents(config)
    assert not parsed.errors

    provider = FakeEmbeddingProvider()
    collections = FakeCollectionFactory()
    indexed = index_chunks(
        config,
        batch_size=4,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert indexed.index_ready

    both = search_vector_chunks(
        config,
        "hidden book secret",
        source_class="both",
        limit=20,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert {result["source_class"] for result in both["results"]} == {
        "canonical",
        "fanfiction",
    }
    assert all(result["spans"] for result in both["results"])

    canonical_work_id = next(
        result["work_id"]
        for result in both["results"]
        if result["source_class"] == "canonical"
    )
    by_work = search_vector_chunks(
        config,
        "hidden book secret",
        source_class="both",
        work_id=canonical_work_id,
        limit=20,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert {result["work_id"] for result in by_work["results"]} == {
        canonical_work_id
    }

    with sqlite3.connect(config.storage_directory / "scriptlab.sqlite3") as connection:
        fanfiction_ids = [
            row[0]
            for row in connection.execute(
                "SELECT c.chunk_id FROM chunks AS c JOIN documents AS d "
                "ON d.document_id = c.document_id WHERE d.source_class = 'fanfiction'"
            )
        ]
        connection.executemany(
            "UPDATE embedding_jobs SET status = 'failed' "
            "WHERE chunk_id = ? AND index_version = ?",
            [(chunk_id, indexed.index_version) for chunk_id in fanfiction_ids],
        )

    incomplete = search_vector_chunks(
        config,
        "hidden book secret",
        source_class="both",
        limit=20,
        embedding_provider=provider,
        collection_factory=collections,
    )
    assert {result["source_class"] for result in incomplete["results"]} == {
        "canonical"
    }
    assert incomplete["coverage"] == {
        "documents_ready": 1,
        "documents_total": 2,
    }
    fts_results = search_chunks(
        config, "hidden book secret", source_class="both", limit=20
    )
    assert {result["source_class"] for result in fts_results} == {
        "canonical",
        "fanfiction",
    }
    hybrid_candidates = search_chunks(
        config,
        "hidden book secret",
        source_class="both",
        limit=20,
        ready_index_version=indexed.index_version,
        require_all_terms=False,
    )
    assert {result["source_class"] for result in hybrid_candidates} == {
        "canonical"
    }


def test_repeated_provider_failures_stop_early_and_remain_resumable(tmp_path):
    config = make_config(tmp_path)
    total = _chunk_count(config)
    collections = FakeCollectionFactory()
    result = index_chunks(
        config,
        batch_size=1,
        embedding_provider=AlwaysFailingEmbeddingProvider(),
        collection_factory=collections,
    )

    assert not result.index_ready
    assert result.chunks_failed == 3
    assert result.chunks_ready == 0
    assert len(result.errors) == 3
    assert total > result.chunks_failed
    status = get_status(config)["vector_index"]
    assert status["status"] == "building"
    assert status["failed_chunks"] == 3
    assert status["pending_chunks"] == total - 3

    resumed = index_chunks(
        config,
        batch_size=2,
        embedding_provider=FakeEmbeddingProvider(),
        collection_factory=collections,
    )
    assert resumed.index_ready
    assert resumed.chunks_ready == total


def test_retry_failed_only_reembeds_only_failed_chunks(tmp_path):
    config = make_config(tmp_path)
    collections = FakeCollectionFactory()
    initial = index_chunks(
        config,
        batch_size=2,
        embedding_provider=FakeEmbeddingProvider(),
        collection_factory=collections,
    )
    collection_name = _index_rows(config)[0][2]
    collection = collections.collections[collection_name]
    with sqlite3.connect(config.storage_directory / "scriptlab.sqlite3") as connection:
        failed_ids = [
            row[0]
            for row in connection.execute(
                "SELECT chunk_id FROM embedding_jobs WHERE index_version = ? "
                "ORDER BY chunk_id LIMIT 2",
                (initial.index_version,),
            )
        ]
        connection.executemany(
            "UPDATE embedding_jobs SET status='failed', error='synthetic outage' "
            "WHERE chunk_id=? AND index_version=?",
            [(chunk_id, initial.index_version) for chunk_id in failed_ids],
        )
    collection.delete(failed_ids)
    provider = FakeEmbeddingProvider()

    retried = index_chunks(
        config,
        batch_size=4,
        retry_failed_only=True,
        embedding_provider=provider,
        collection_factory=collections,
    )

    assert retried.index_version == initial.index_version
    assert retried.index_ready
    assert retried.chunks_indexed_this_run == len(failed_ids)
    assert provider.calls == 1
    assert collection.count() == _chunk_count(config)


def test_ollama_provider_retries_internal_tokenizer_connection_errors(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"embeddings": [[0.1, 0.2]]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                BytesIO(b'{"error":"/tokenize connectex temporary failure"}'),
            )
        return Response()

    monkeypatch.setattr(vector_module, "urlopen", fake_urlopen)
    provider = OllamaEmbeddingProvider("test-model", "http://localhost:11434")

    assert provider.embed(["small synthetic input"]) == [[0.1, 0.2]]
    assert len(calls) == 2


def test_ollama_provider_does_not_retry_permanent_bad_requests(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        raise HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":"input exceeds model context"}'),
        )

    monkeypatch.setattr(vector_module, "urlopen", fake_urlopen)
    provider = OllamaEmbeddingProvider("test-model", "http://localhost:11434")

    with pytest.raises(VectorIndexError, match="input exceeds model context"):
        provider.embed(["synthetic input"])
    assert len(calls) == 1


def test_ollama_provider_splits_transiently_failing_batches(monkeypatch):
    batch_sizes = []

    class Response:
        def __init__(self, texts):
            self.texts = texts

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(
                {"embeddings": [[float(len(text))] for text in self.texts]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        texts = json.loads(request.data.decode("utf-8"))["input"]
        batch_sizes.append(len(texts))
        if len(texts) > 1:
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                BytesIO(b'{"error":"/tokenize connectex temporary failure"}'),
            )
        return Response(texts)

    monkeypatch.setattr(vector_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(vector_module.time, "sleep", lambda seconds: None)
    provider = OllamaEmbeddingProvider("test-model", "http://localhost:11434")

    assert provider.embed(["a", "bb", "ccc", "dddd"]) == [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]
    assert batch_sizes[0:3] == [4, 4, 4]
    assert 1 in batch_sizes