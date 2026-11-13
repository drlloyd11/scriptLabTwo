"""Score preservation and retrieval of manually labeled screenplay actions."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict

from scriptlab.config import load_config
from scriptlab.database import search_chunks, validate_database


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _find_evidence(text: str, evidence_text: str):
    words = evidence_text.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    return re.search(pattern, text, flags=re.IGNORECASE)


def evaluate(config_path: Path, actions_path: Path, output_path: Path) -> Dict[str, object]:
    config = load_config(config_path)
    with actions_path.open("r", encoding="utf-8") as file:
        audit = json.load(file)
    database_path = config.storage_directory / "scriptlab.sqlite3"
    import sqlite3

    results = []
    for action in audit["queries"]:
        with sqlite3.connect(database_path) as connection:
            connection.row_factory = sqlite3.Row
            document = connection.execute(
                "SELECT document_id, source_class, normalized_text FROM documents "
                "WHERE filename = ?",
                (action["expected_filename"],),
            ).fetchone()
            if document is None:
                raise RuntimeError("Missing document: {}".format(action["expected_filename"]))
            if document["document_id"] != action["expected_document_id"]:
                raise RuntimeError("Document ID changed for {}".format(action["id"]))
            if document["source_class"] != action["source"]:
                raise RuntimeError("Source class changed for {}".format(action["id"]))
            source_match = _find_evidence(
                document["normalized_text"], action["evidence_text"]
            )
            source_preserved = source_match is not None
            chunks = connection.execute(
                "SELECT chunk_id, chunk_type, chunk_text FROM chunks "
                "WHERE document_id = ?",
                (document["document_id"],),
            ).fetchall()
            scene_rows = [
                row for row in chunks
                if row["chunk_type"] == "scene"
                and _find_evidence(row["chunk_text"], action["evidence_text"])
            ]
            cited_scene_preserved = bool(scene_rows)
            cited_spans_cover_evidence = False
            if source_match is not None:
                for scene in scene_rows:
                    spans = connection.execute(
                        "SELECT start_offset, end_offset FROM chunk_spans "
                        "WHERE chunk_id = ? ORDER BY span_ordinal",
                        (scene["chunk_id"],),
                    ).fetchall()
                    if any(
                        span["start_offset"] <= source_match.start()
                        and source_match.end() <= span["end_offset"]
                        for span in spans
                    ):
                        cited_spans_cover_evidence = True
                        break

            turns = connection.execute(
                "SELECT original_speaker_label, turn_text FROM dialogue_turns "
                "WHERE document_id = ?",
                (document["document_id"],),
            ).fetchall()
            dialogue_contamination = []
            cue_annotations = []
            for turn in turns:
                if _find_evidence(turn["turn_text"], action["evidence_text"]):
                    if _find_evidence(
                        turn["original_speaker_label"] or "", action["evidence_text"]
                    ):
                        cue_annotations.append(turn["original_speaker_label"])
                    else:
                        lines = [line.strip() for line in turn["turn_text"].splitlines() if line.strip()]
                        opening_parenthetical = next(
                            (
                                line
                                for line in lines[1:]
                                if line.startswith("(") and line.endswith(")")
                            ),
                            "",
                        )
                        if _find_evidence(opening_parenthetical, action["evidence_text"]):
                            cue_annotations.append(turn["original_speaker_label"])
                        else:
                            dialogue_contamination.append(turn["original_speaker_label"])

        fts_results = search_chunks(
            config,
            action["query"],
            source_class=action["source"],
            limit=audit.get("limit", 5),
        )
        scene_ids = {row["chunk_id"] for row in scene_rows}
        fts_hit = any(item["chunk_id"] in scene_ids for item in fts_results)
        results.append(
            {
                "id": action["id"],
                "expected_filename": action["expected_filename"],
                "source_preserved": source_preserved,
                "scene_chunk_preserved": cited_scene_preserved,
                "citation_span_covers_action": cited_spans_cover_evidence,
                "scene_chunk_ids": sorted(scene_ids),
                "dialogue_contamination": bool(dialogue_contamination),
                "contaminating_speakers": dialogue_contamination,
                "cue_action_annotations": cue_annotations,
                "fts_hit_at_5": fts_hit,
                "fts_top_filenames": [item["filename"] for item in fts_results],
            }
        )

    total = len(results)
    validation = validate_database(config)
    report = {
        "version": audit["version"],
        "questions": total,
        "data_preservation": sum(item["source_preserved"] for item in results),
        "scene_chunk_preservation": sum(item["scene_chunk_preserved"] for item in results),
        "citation_span_coverage": sum(item["citation_span_covers_action"] for item in results),
        "dialogue_contamination": sum(item["dialogue_contamination"] for item in results),
        "cue_action_annotations": sum(bool(item["cue_action_annotations"]) for item in results),
        "fts_hits_at_5": sum(item["fts_hit_at_5"] for item in results),
        "citation_database_valid": validation["valid"],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate(args.config, args.actions, args.output)
    total = report["questions"]
    print(
        "source={}/{} scene={}/{} spans={}/{} dialogue_contamination={}/{} "
        "FTS_Hit@5={}/{} citations_valid={}".format(
            report["data_preservation"],
            total,
            report["scene_chunk_preservation"],
            total,
            report["citation_span_coverage"],
            total,
            report["dialogue_contamination"],
            total,
            report["fts_hits_at_5"],
            total,
            report["citation_database_valid"],
        )
    )
    print("Results: {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
