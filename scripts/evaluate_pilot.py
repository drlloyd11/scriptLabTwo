"""Evaluate frozen ScriptLab retrieval questions without copying source text."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from scriptlab.cli import _reciprocal_rank_fusion
from scriptlab.config import load_config
from scriptlab.database import search_chunks
from scriptlab.vectors import search_vector_chunks


def evaluate(config_path: Path, queries_path: Path, output_path: Path) -> Dict[str, object]:
    config = load_config(config_path)
    with queries_path.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    report: Dict[str, object] = {
        "benchmark_version": benchmark["version"],
        "limit": benchmark["limit"],
        "modes": {},
        "queries": [],
    }
    mode_hits = {mode: 0 for mode in ("fts", "semantic", "hybrid")}

    total_questions = len(benchmark["queries"])
    for question_number, question in enumerate(benchmark["queries"], start=1):
        print(
            "[{}/{}] {}: starting".format(
                question_number, total_questions, question["id"]
            ),
            flush=True,
        )
        limit = benchmark["limit"]
        fts_results = search_chunks(
            config,
            question["query"],
            source_class=question["source"],
            work_id=question.get("work_id"),
            limit=max(limit * 5, limit),
        )
        print("[{}/{}] {}: FTS ready".format(question_number, total_questions, question["id"]), flush=True)
        vector_result = search_vector_chunks(
            config,
            question["query"],
            source_class=question["source"],
            work_id=question.get("work_id"),
            limit=max(limit * 10, limit),
        )
        print("[{}/{}] {}: semantic ready".format(question_number, total_questions, question["id"]), flush=True)
        hybrid_fts = search_chunks(
            config,
            question["query"],
            source_class=question["source"],
            work_id=question.get("work_id"),
            limit=max(limit * 10, limit),
            ready_index_version=vector_result["index_version"],
            require_all_terms=False,
        )
        print("[{}/{}] {}: hybrid candidates ready".format(question_number, total_questions, question["id"]), flush=True)
        mode_results = {
            "fts": fts_results[:limit],
            "semantic": vector_result["results"][:limit],
            "hybrid": _reciprocal_rank_fusion(
                hybrid_fts, vector_result["results"], limit
            ),
        }
        evaluated_modes = {}
        expected_ids = set(question["expected_chunk_ids"])
        for mode, results in mode_results.items():
            hit = any(result["chunk_id"] in expected_ids for result in results)
            mode_hits[mode] += int(hit)
            evaluated_modes[mode] = {
                "hit_at_5": hit,
                "top_filenames": [result["filename"] for result in results],
                "top_chunk_ids": [result["chunk_id"] for result in results],
            }
        report["queries"].append(
            {
                "id": question["id"],
                "query": question["query"],
                "source": question["source"],
                "expected_filename": question["expected_filename"],
                "expected_document_id": question["expected_document_id"],
                "expected_chunk_ids": question["expected_chunk_ids"],
                "modes": evaluated_modes,
                "coverage": vector_result["coverage"],
            }
        )
        query_hits = ", ".join(
            mode for mode, result in evaluated_modes.items() if result["hit_at_5"]
        ) or "none"
        print(
            "[{}/{}] {}: complete; Hit@5 modes={}".format(
                question_number, total_questions, question["id"], query_hits
            ),
            flush=True,
        )

    total = len(benchmark["queries"])
    report["modes"] = {
        mode: {"hits_at_5": hits, "questions": total, "rate": hits / total}
        for mode, hits in mode_hits.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate(args.config, args.queries, args.output)
    print("Questions: {}".format(len(report["queries"])))
    for mode, result in report["modes"].items():
        print(
            "{} Hit@5: {}/{} ({:.0%})".format(
                mode,
                result["hits_at_5"],
                result["questions"],
                result["rate"],
            )
        )
    print("Results: {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
