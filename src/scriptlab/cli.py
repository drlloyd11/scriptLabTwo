"""Command-line entry point for the ScriptLab prototype."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from scriptlab.config import ConfigError, load_config, write_sample_config
from scriptlab.corpus import (
    CorpusError,
    create_inventory_report,
    preview_source_file,
)
from scriptlab.database import (
    PersistenceError,
    get_status,
    import_sources,
    inspect_document,
    parse_documents,
    search_chunks,
    validate_database,
)
from scriptlab.vectors import VectorIndexError, index_chunks, search_vector_chunks


def _reciprocal_rank_fusion(fts_results, vector_results, limit: int, rank_constant: int = 40):
    scores = {}
    results_by_id = {}
    for ranked_results in (fts_results, vector_results):
        for rank, result in enumerate(ranked_results, start=1):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            results_by_id.setdefault(chunk_id, result)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    results = []
    for chunk_id in ordered[:limit]:
        result = dict(results_by_id[chunk_id])
        result["rrf_score"] = scores[chunk_id]
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scriptlab",
        description="Local, evidence-first search for scripts and fanfiction.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scriptlab.yaml"),
        help="configuration file (default: scriptlab.yaml)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser(
        "init", help="write a sample configuration without creating source folders"
    )
    init_parser.add_argument(
        "--output",
        type=Path,
        default=Path("scriptlab.yaml"),
        help="path for the new configuration (default: scriptlab.yaml)",
    )

    commands.add_parser("inventory", help="inventory and extract source folders")

    preview_parser = commands.add_parser(
        "preview", help="preview extracted text from one configured source file"
    )
    preview_parser.add_argument("file", type=Path, help="path to a direct-child source file")
    preview_parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="maximum extracted characters to display (default: 4000)",
    )

    commands.add_parser("ingest", help="extract and store configured sources")
    index_parser = commands.add_parser(
        "index", help="resume local vector indexing for stored chunks"
    )
    index_parser.add_argument("--batch-size", type=int, default=16)
    index_parser.add_argument(
        "--limit", type=int, help="maximum new chunk embeddings for this run"
    )
    index_parser.add_argument(
        "--rebuild", action="store_true", help="build a replacement index safely"
    )
    index_parser.add_argument(
        "--retry-failed", action="store_true", help="retry only failed chunk jobs"
    )
    commands.add_parser("status", help="show database counts and per-file issues")
    commands.add_parser("validate", help="validate FTS and chunk citations")
    parse_parser = commands.add_parser("parse", help="parse stored documents into scenes and chunks")
    parse_parser.add_argument(
        "document_id", nargs="?", help="parse only this document (default: all documents)"
    )
    inspect_parser = commands.add_parser(
        "inspect", help="inspect parsed scenes, speakers, spans, and warnings"
    )
    inspect_parser.add_argument("document_id", help="stored document ID")

    search_parser = commands.add_parser("search", help="search the local corpus")
    search_parser.add_argument("query", help="search query")
    search_parser.add_argument(
        "--mode", choices=("fts", "semantic", "hybrid"), default="fts"
    )
    search_parser.add_argument(
        "--source",
        choices=("canonical", "fanfiction", "both"),
        default="canonical",
    )
    search_parser.add_argument("--work-id")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument(
        "--fallback-to-fts",
        action="store_true",
        help="explicitly use keyword search if semantic search is unavailable",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        try:
            config_path = write_sample_config(args.output)
        except ConfigError as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        print("Created sample configuration: {}".format(config_path))
        print("Edit its source paths; no source directories were created.")
        return 0

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))

    if args.command == "inventory":
        try:
            report_path, report = create_inventory_report(config)
        except CorpusError as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2

        summary = report["summary"]
        print("Inventory report: {}".format(report_path))
        print(
            "Supported: {}; ready: {}; recovered: {}; failed: {}; unsupported: {}".format(
                summary["supported_files"],
                summary["ready_files"],
                summary["recovered_files"],
                summary["failed_files"],
                summary["unsupported_files"],
            )
        )
        for item in report["files"]:
            for message in item["recovery_warnings"]:
                print(
                    "Recovery warning: {}: {}".format(item["path"], message),
                    file=sys.stderr,
                )
            if item["status"] == "failed":
                print(
                    "Failed: {} ({})".format(item["path"], item["error"]),
                    file=sys.stderr,
                )
        for item in report["unsupported"]:
            print(
                "Unsupported: {} ({})".format(item["path"], item["format"])
            )
        return 1 if summary["failed_files"] else 0

    if args.command == "preview":
        if args.max_chars < 1:
            parser.error("--max-chars must be greater than zero.")
        try:
            recovery_warnings = []
            source_class, encoding, digest, text = preview_source_file(
                config, args.file, recovery_warnings=recovery_warnings
            )
        except CorpusError as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        for message in recovery_warnings:
            print("Recovery warning: {}".format(message), file=sys.stderr)
        print(
            "Source class: {}\nEncoding: {}\nSHA-256: {}\n---".format(
                source_class, encoding, digest
            )
        )
        print(text[: args.max_chars])
        if len(text) > args.max_chars:
            print(
                "\n[Preview truncated at {} characters.]".format(args.max_chars),
                file=sys.stderr,
            )
        return 0

    if args.command == "ingest":
        try:
            result = import_sources(config)
        except (CorpusError, PersistenceError) as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        print(
            "Imported: {}; unchanged: {}; conflicts: {}; failed: {}; unsupported: {}".format(
                result.added,
                result.unchanged,
                result.conflicts,
                result.failed,
                result.unsupported,
            )
        )
        return 1 if result.conflicts or result.failed else 0

    if args.command == "index":
        if args.batch_size < 1:
            parser.error("--batch-size must be greater than zero.")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be greater than zero.")
        if args.retry_failed and args.rebuild:
            parser.error("--retry-failed cannot be combined with --rebuild.")
        is_terminal = sys.stdout.isatty()
        last_progress_at = 0.0
        last_progress_bucket = -1
        last_progress_values = [0, 0, 0, 0, 0, 0]

        def report_index_progress(
            scanned, scan_total, ready, indexed, reused, failed, force=False
        ):
            nonlocal last_progress_at, last_progress_bucket
            last_progress_values[:] = [
                scanned,
                scan_total,
                ready,
                indexed,
                reused,
                failed,
            ]
            now = time.monotonic()
            scan_percent = (
                100.0
                if not scan_total
                else min(100.0, scanned * 100.0 / scan_total)
            )
            if is_terminal:
                if not force and now - last_progress_at < 0.5:
                    return
                bar_width = 28
                filled = int(bar_width * scan_percent / 100.0)
                bar = "#" * filled + "-" * (bar_width - filled)
                print(
                    "\rIndex scan [{}] {:5.1f}% | checked {:,}/{:,} | ready {:,} | "
                    "embedded {:,} | reused {:,} | failed {:,}".format(
                        bar,
                        scan_percent,
                        scanned,
                        scan_total,
                        ready,
                        indexed,
                        reused,
                        failed,
                    ),
                    end="",
                    flush=True,
                )
                last_progress_at = now
                return

            progress_bucket = int(scan_percent // 5)
            if not force and progress_bucket <= last_progress_bucket:
                return
            print(
                "Index scan: {:5.1f}% ({:,}/{:,} checked); ready {:,}; embedded {:,}; "
                "reused {:,}; failed {:,}".format(
                    scan_percent, scanned, scan_total, ready, indexed, reused, failed
                ),
                flush=True,
            )
            last_progress_bucket = progress_bucket

        try:
            result = index_chunks(
                config,
                batch_size=args.batch_size,
                limit=args.limit,
                rebuild=args.rebuild,
                retry_failed_only=args.retry_failed,
                progress_callback=report_index_progress,
            )
        except (PersistenceError, VectorIndexError, ValueError) as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        report_index_progress(
            *last_progress_values,
            force=True,
        )
        if is_terminal:
            print()
        print(
            "Vector index {} model={!r}: indexed {}; reused {}; ready {}/{}; "
            "failed {}; documents ready {}/{}".format(
                "ready" if result.index_ready else "building",
                result.embedding_model,
                result.chunks_indexed_this_run,
                result.chunks_reused,
                result.chunks_ready,
                result.chunks_expected,
                result.chunks_failed,
                result.documents_ready,
                get_status(config)["documents"],
            )
        )
        for error in result.errors:
            print("Index warning: {}".format(error), file=sys.stderr)
        return 1 if result.errors else 0

    if args.command == "status":
        try:
            status = get_status(config)
        except PersistenceError as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        print("Database: {}".format(status["database"]))
        print("Works: {}; documents: {}".format(status["works"], status["documents"]))
        vector = status["vector_index"]
        if vector["status"] == "not_built":
            print(
                "Vector index: not built; pending chunks: {}; documents ready: 0/{}".format(
                    vector["pending_chunks"], vector["documents_with_chunks"]
                )
            )
        else:
            print(
                "Vector index: {} model={!r}; chunks ready: {}/{}; pending: {}; "
                "failed: {}; documents ready: {}/{}".format(
                    vector["status"],
                    vector["embedding_model"],
                    vector["ready_chunks"],
                    vector["expected_chunks"],
                    vector["pending_chunks"],
                    vector["failed_chunks"],
                    vector["documents_ready"],
                    vector["documents_with_chunks"],
                )
            )
        for issue in status["issues"]:
            print(
                "{}: {} [{}] {}".format(
                    issue["source_path"],
                    issue["status"],
                    issue["source_class"],
                    issue["message"],
                ),
                file=sys.stderr if issue["status"] != "unsupported" else sys.stdout,
            )
        return 1 if any(issue["status"] in ("failed", "conflict") for issue in status["issues"]) else 0

    if args.command == "search":
        if args.limit < 1:
            parser.error("--limit must be greater than zero.")
        if args.mode == "fts" and args.fallback_to_fts:
            parser.error("--fallback-to-fts applies only to semantic or hybrid search.")

        actual_mode = args.mode
        coverage = None
        if args.mode == "fts":
            try:
                results = search_chunks(
                    config,
                    args.query,
                    source_class=args.source,
                    work_id=args.work_id,
                    limit=args.limit,
                )
            except (PersistenceError, ValueError) as exc:
                print("scriptlab: error: {}".format(exc), file=sys.stderr)
                return 2
        else:
            try:
                vector_result = search_vector_chunks(
                    config,
                    args.query,
                    source_class=args.source,
                    work_id=args.work_id,
                    limit=(
                        args.limit
                        if args.mode == "semantic"
                        else max(args.limit * 10, args.limit)
                    ),
                )
            except (PersistenceError, VectorIndexError, ValueError) as exc:
                if not args.fallback_to_fts:
                    print("scriptlab: error: {}".format(exc), file=sys.stderr)
                    return 2
                print(
                    "Vector search unavailable: {}; using FTS by explicit request.".format(
                        exc
                    ),
                    file=sys.stderr,
                )
                actual_mode = "fts"
                try:
                    results = search_chunks(
                        config,
                        args.query,
                        source_class=args.source,
                        work_id=args.work_id,
                        limit=args.limit,
                    )
                except (PersistenceError, ValueError) as fts_error:
                    print("scriptlab: error: {}".format(fts_error), file=sys.stderr)
                    return 2
            else:
                coverage = vector_result["coverage"]
                vector_results = vector_result["results"]
                if args.mode == "semantic":
                    results = vector_results[: args.limit]
                else:
                    try:
                        fts_results = search_chunks(
                            config,
                            args.query,
                            source_class=args.source,
                            work_id=args.work_id,
                            limit=max(args.limit * 10, args.limit),
                            ready_index_version=vector_result["index_version"],
                            require_all_terms=False,
                        )
                    except (PersistenceError, ValueError) as exc:
                        print("scriptlab: error: {}".format(exc), file=sys.stderr)
                        return 2
                    results = _reciprocal_rank_fusion(
                        fts_results, vector_results, args.limit
                    )

        print(
            "Search mode: {}; source: {}; results: {}; requested_mode={}; actual_mode={}".format(
                args.mode, args.source, len(results), args.mode, actual_mode
            )
        )
        if coverage is not None:
            print(
                "Vector coverage: {}/{} imported documents fully ready; other documents are excluded from semantic/hybrid results.".format(
                    coverage["documents_ready"], coverage["documents_total"]
                )
            )
        for result in results:
            print(
                "\n{} [{}] type={} work={}".format(
                    result["filename"],
                    result["source_class"],
                    result["chunk_type"],
                    result["work_id"],
                )
            )
            print("Chunk: {}".format(result["chunk_id"]))
            print(
                "Source spans: {}".format(
                    ", ".join(
                        "[{}:{})".format(span["start_offset"], span["end_offset"])
                        for span in result["spans"]
                    )
                )
            )
            if result["speaker_labels"]:
                print("Speakers: {}".format(", ".join(result["speaker_labels"])))
            if result["scene_headings"]:
                print("Scenes: {}".format("; ".join(result["scene_headings"])))
            print(result["chunk_text"])
        return 0

    if args.command == "validate":
        try:
            report = validate_database(config)
        except PersistenceError as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        print(
            "Documents: {}; chunks: {}; FTS rows: {}".format(
                report["documents"], report["chunks"], report["fts_rows"]
            )
        )
        if report["valid"]:
            print("FTS integrity and chunk citations: valid")
            return 0
        for error in report["errors"]:
            print("Validation error: {}".format(error), file=sys.stderr)
        return 1

    if args.command == "parse":
        try:
            summary = parse_documents(config, args.document_id)
        except PersistenceError as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        print(
            "Parsed documents: {}; scenes: {}; dialogue turns: {}; chunks: {}; warnings: {}".format(
                summary.documents_parsed,
                summary.scenes,
                summary.dialogue_turns,
                summary.chunks,
                summary.warnings,
            )
        )
        for error in summary.errors:
            print("Parse failed: {}".format(error), file=sys.stderr)
        return 1 if summary.errors else 0

    if args.command == "inspect":
        try:
            report = inspect_document(config, args.document_id)
        except PersistenceError as exc:
            print("scriptlab: error: {}".format(exc), file=sys.stderr)
            return 2
        print(
            "Document: {} [{}] {}".format(
                report["document_id"], report["source_class"], report["filename"]
            )
        )
        for scene in report["scenes"]:
            print(
                "Scene {} [{}:{}) {}".format(
                    scene["scene_ordinal"],
                    scene["start_offset"],
                    scene["end_offset"],
                    scene["scene_heading"] or "(no heading)",
                )
            )
        for turn in report["dialogue_turns"]:
            speaker = turn["original_speaker_label"] or "(unassigned)"
            resolved = turn["speaker_character_id"] or "unresolved"
            print(
                "Turn {} [{}:{}) speaker={} resolved={} status={}".format(
                    turn["turn_ordinal"],
                    turn["start_offset"],
                    turn["end_offset"],
                    speaker,
                    resolved,
                    turn["attribution_status"],
                )
            )
            print("  Source: {!r}".format(turn["source_snippet"][:400]))
        for chunk in report["chunks"]:
            print(
                "Chunk {} type={} version={} ordinal={}".format(
                    chunk["chunk_id"],
                    chunk["chunk_type"],
                    chunk["chunker_version"],
                    chunk["chunk_ordinal"],
                )
            )
            for span in chunk["spans"]:
                print(
                    "  Span {} [{}:{}) {!r}".format(
                        span["span_ordinal"],
                        span["start_offset"],
                        span["end_offset"],
                        span["source_snippet"][:400],
                    )
                )
        for warning in report["warnings"]:
            print(
                "Warning {} [{}:{}) {} {!r}".format(
                    warning["warning_code"],
                    warning["start_offset"],
                    warning["end_offset"],
                    warning["message"],
                    warning["source_snippet"],
                )
            )
        return 0

    print(
        "scriptlab: {!r} is not implemented yet.".format(args.command),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
