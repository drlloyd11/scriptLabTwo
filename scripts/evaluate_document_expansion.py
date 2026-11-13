"""Experiment with per-document Chroma candidate expansion and local reranking."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sentence_transformers import CrossEncoder

from scriptlab.cli import _reciprocal_rank_fusion
from scriptlab.config import load_config
from scriptlab.database import get_status, search_chunks, search_chunks_by_ids
from scriptlab.vectors import (
    ChromaVectorCollection,
    OllamaEmbeddingProvider,
    search_vector_chunks,
)


def evaluate(
    config_path: Path,
    queries_path: Path,
    output_path: Path,
    model_name: str,
    global_candidates: int,
    documents_per_query: int,
    chunks_per_document: int,
    rerank_limit: int,
) -> Dict[str, object]:
    config = load_config(config_path)
    with queries_path.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    index = get_status(config)["vector_index"]
    if index["status"] != "active":
        raise RuntimeError("A complete active vector index is required.")
    collection = ChromaVectorCollection(
        config,
        index["collection_name"],
        index["embedding_model"],
        index["index_version"],
    )
    embedder = OllamaEmbeddingProvider(config.embedding_model, config.ollama_url)
    reranker = CrossEncoder(model_name, device="cpu")
    output: Dict[str, object] = {
        "benchmark_version": benchmark["version"],
        "embedding_model": config.embedding_model,
        "reranker_model": model_name,
        "global_candidates": global_candidates,
        "documents_per_query": documents_per_query,
        "chunks_per_document": chunks_per_document,
        "rerank_limit": rerank_limit,
        "queries": [],
    }

    for question in benchmark["queries"]:
        vector = search_vector_chunks(
            config,
            question["query"],
            source_class=question["source"],
            work_id=question.get("work_id"),
            limit=global_candidates,
        )
        lexical = search_chunks(
            config,
            question["query"],
            source_class=question["source"],
            work_id=question.get("work_id"),
            limit=global_candidates,
            ready_index_version=vector["index_version"],
            require_all_terms=False,
        )
        baseline = _reciprocal_rank_fusion(
            lexical, vector["results"], global_candidates
        )
        selected_documents: List[str] = []
        for item in baseline:
            if item["document_id"] not in selected_documents:
                selected_documents.append(item["document_id"])
            if len(selected_documents) == documents_per_query:
                break

        query_embedding = embedder.embed([question["query"]])[0]
        expanded_ids: List[str] = []
        for document_id in selected_documents:
            expanded_ids.extend(
                chunk_id
                for chunk_id, _ in collection.query(
                    query_embedding,
                    chunks_per_document,
                    where={"document_id": {"$eq": document_id}},
                )
            )

        all_ids = list(
            dict.fromkeys(
                [item["chunk_id"] for item in baseline] + expanded_ids
            )
        )
        expanded = search_chunks_by_ids(
            config,
            all_ids,
            question["source"],
            question.get("work_id"),
            vector["index_version"],
        )
        scores = reranker.predict(
            [(question["query"], item["chunk_text"]) for item in expanded],
            batch_size=16,
            show_progress_bar=False,
        )
        reranked = [
            item
            for _, item in sorted(
                zip(scores, expanded), key=lambda pair: -float(pair[0])
            )
        ]
        expected_ids = set(question["expected_chunk_ids"])
        baseline_top = baseline[:rerank_limit]
        reranked_top = reranked[:rerank_limit]
        output["queries"].append(
            {
                "id": question["id"],
                "query": question["query"],
                "expected_filename": question["expected_filename"],
                "expected_chunk_ids": question["expected_chunk_ids"],
                "selected_document_ids": selected_documents,
                "expanded_candidates": len(expanded),
                "baseline_hit_at_5": any(
                    item["chunk_id"] in expected_ids for item in baseline_top
                ),
                "expanded_reranked_hit_at_5": any(
                    item["chunk_id"] in expected_ids for item in reranked_top
                ),
                "baseline_top_chunk_ids": [
                    item["chunk_id"] for item in baseline_top
                ],
                "expanded_reranked_top_chunk_ids": [
                    item["chunk_id"] for item in reranked_top
                ],
                "expanded_reranked_top_filenames": [
                    item["filename"] for item in reranked_top
                ],
            }
        )

    total = len(output["queries"])
    baseline_hits = sum(
        item["baseline_hit_at_5"] for item in output["queries"]
    )
    expanded_hits = sum(
        item["expanded_reranked_hit_at_5"] for item in output["queries"]
    )
    output["summary"] = {
        "questions": total,
        "baseline_hits_at_5": baseline_hits,
        "expanded_reranked_hits_at_5": expanded_hits,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--global-candidates", type=int, default=50)
    parser.add_argument("--documents-per-query", type=int, default=5)
    parser.add_argument("--chunks-per-document", type=int, default=50)
    parser.add_argument("--rerank-limit", type=int, default=5)
    args = parser.parse_args()

    for name in (
        "global_candidates",
        "documents_per_query",
        "chunks_per_document",
        "rerank_limit",
    ):
        if getattr(args, name) < 1:
            parser.error("--{} must be greater than zero.".format(name.replace("_", "-")))

    result = evaluate(
        args.config,
        args.queries,
        args.output,
        args.model,
        args.global_candidates,
        args.documents_per_query,
        args.chunks_per_document,
        args.rerank_limit,
    )
    summary = result["summary"]
    print(
        "Baseline Hit@5: {}/{}; expanded+rereanked Hit@5: {}/{}".format(
            summary["baseline_hits_at_5"],
            summary["questions"],
            summary["expanded_reranked_hits_at_5"],
            summary["questions"],
        )
    )
    print("Results: {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
