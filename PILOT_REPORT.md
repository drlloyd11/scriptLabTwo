# ScriptLab 20-Document Pilot

**Run date:** 2026-11-11  
**Corpus:** 5 Buffy HTML scripts, 5 numbered Angel TXT scripts, and 10 Kitten Board fanfiction TXT files. Angel HTML alternates were excluded, following the recorded source-scope decision. The sample and database are isolated under ignored `data/pilot_v1/`; the existing 983-document database and vector index were not modified.

## Inventory and Parsing

- Inventory: 20 supported, 20 ready, 0 failed, 0 unsupported; 3 TXT files were recovered with control/undefined-byte warnings. A failing Buffy HTML candidate was replaced in the pilot copy before ingestion.
- Parsed: 20 documents, 568 scenes, 4,292 dialogue turns, 4,865 chunks, 25 warnings (13 `ambiguous_speaker`, 12 `unmatched_dialogue`). All warnings were in canonical scripts; fanfiction speaker attribution remained disabled.
- Validation: 4,865 chunks and 4,865 FTS rows; FTS integrity and citation reconstruction passed.
- Vector indexing: 4,865/4,865 chunks indexed with `qwen3-embedding:0.6b`; all 20 documents vector-ready.

Manual review confirmed useful HTML/TXT text extraction and visible speaker labels. Before the parser fix, 6 of the first 9 turns reviewed across Buffy `1x01.htm`, Buffy `2x05.htm`, and Angel `1x11.txt` included obvious action or camera-description text. This convenience sample is not a corpus-wide error-rate estimate. `KATE (0.S.)` remains as written in the source; it may be an OCR/source typo and was not silently normalized.

## Angel Clean-Copy Reparse

Following the recovery warnings, a separate working copy was created under `data/pilot_v1_parserfix_clean/`; original Angel files were not edited. The cleanup removed only the previously reported bytes: 94 NUL, 1 `0x02`, and 46 `0x03` bytes from `1x02.txt`; 106 NUL, 3 `0x02`, and 50 `0x03` bytes from `3x05.txt`; and one undefined Windows-1252 `0x8D` byte from `4x22.txt`. Before/after hashes and removal counts are recorded in `data/pilot_v1_parserfix_clean/sanitization_manifest.json`.

The clean copy inventoried as 20 ready, 0 recovered, 0 failed. It was ingested and reparsed into fresh storage: 20 documents, 568 scenes, 4,291 dialogue turns, and 4,845 chunks; citation validation passed. All 20 normalized texts exactly match the earlier parser-fix database, and the clean database has no persisted extraction-recovery warnings. The three original Angel files' SHA-256 hashes still match the manifest's pre-clean hashes. No vector index was rebuilt because the normalized text is identical; all prior indexes remain untouched.

## Retrieval Evaluation

The frozen v1 set has 10 questions and 11 manually identified relevant chunk IDs. Its original results remain in `data/pilot_v1/results.json`; the same queries were rerun after hybrid lexical matching changed, with results in `data/pilot_v1/results_after_hybrid_or.json`.

| Benchmark/run | FTS Hit@5 | Semantic Hit@5 | Hybrid Hit@5 |
|---|---:|---:|---:|
| v1 before hybrid OR candidates | 0/10 | 2/10 | 2/10 |
| v1 after hybrid OR candidates | 0/10 | 2/10 | 3/10 |
| v2 after manual label audit | 0/10 | 6/10 | 7/10 |
| v3 parser-fix baseline (5 vector / 25 FTS, k=60) | 0/10 | 5/10 | 6/10 |
| v3 balanced candidates (50 vector / 50 FTS, k=40) | 0/10 | 5/10 | 7/10 |
| v3 with Nomic embeddings (50 / 50, k=40) | 0/10 | 3/10 | 7/10 |
| v3 with Mxbai embeddings (50 / 50, k=40) | 0/10 | 6/10 | 8/10 |

V2 is a separately versioned relevance audit, not a replacement for v1: it expands relevant passage labels after manual review and clarifies Q07. Its questions and labels are in `data/pilot_v1/queries_v2.json`; its results are in `data/pilot_v1/results_v2.json`. All 10 expected source documents appeared in hybrid top five on v2, but only 7/10 had a directly relevant labeled passage in the top five. The plan's 8/10 target was not met. The three v2 hybrid passage misses are Q04, Q05, and Q08.

The initial strict FTS mode returned no hits for this conversational-question set because it requires every query term. Standalone FTS remains strict AND; hybrid now uses OR lexical candidates before rank fusion, improving v1 from 2/10 to 3/10 and v2 hybrid to 7/10. This benchmark does not justify changing the frozen queries or claiming retrieval quality beyond the measured sample.

The separate ten-question holdout baseline with Qwen scored FTS 0/10, semantic 1/10, and hybrid 4/10. It was measured before the lexical-weight sweep and was not used to select a setting. Lexical-weight sweeps on v3 did not help: equal weights scored 7/10; lexical weights of 1.25, 1.5, or 2 scored 6/10; weight 3 scored 5/10. No weighted lexical change was adopted. The isolated Nomic index tied Qwen on v3 hybrid but had lower semantic-only recall (3/10 versus 5/10), so it is not a clear replacement; it was not run against the holdout.

Mxbai was tested in a separate index using the same parser-fix corpus and balanced hybrid settings. It scored 8/10 hybrid and 6/10 semantic on v3, improving on Qwen's 7/10 and 5/10. On the untouched holdout, Mxbai scored hybrid 5/10 and semantic 5/10 versus Qwen's 4/10 and 1/10. All ten expected holdout documents appeared in Mxbai's top five, but only five directly relevant passages did; H05 was the only additional passage hit. This is a promising embedding improvement, but it does not meet the pilot's 8/10 goal on holdout. The isolated Mxbai database and outputs are under `data/pilot_v1_parserfix_mxbai/`; the active ScriptLab model remains unchanged.

## Reranker and Query-Rewrite Diagnostics

A local `cross-encoder/ms-marco-MiniLM-L-6-v2` reranked the top 50 fused candidates. It scored 6/10 on v3 versus 7/10 for the unchanged RRF order: Q04 improved to rank 2, but Q01 and Q02 fell below five, Q05 remained at rank 19, and Q08 was not in the candidate pool. It was not adopted and was not evaluated on the holdout. `sentence-transformers` is installed in the local `.venv` for this experiment only; it is not a project dependency and is not wired into search.

Manual Q05 source review found both a scene chunk and a dialogue-exchange chunk that contain the Valet's condition. In the Qwen candidate lists they ranked at vector positions 13/18 and OR-FTS positions 22/27, respectively; neither was in the final top five. The diagnostic manifest is `data/pilot_v1_parserfix/query_q05_label_audit.json`.

Three source-grounded Q08 rewrites were tested separately. The best, “What does Amy threaten to do with magic?”, retrieved the labeled chunk at vector rank 9 and OR-FTS rank 15; it ranked 12 after RRF and 6 after the cross-encoder. No rewrite reached top five. These diagnostics are in `data/pilot_v1_parserfix/query_q08_variants.json` and are excluded from benchmark scores. This points to a candidate recall/query representation issue that simple reranking does not solve.

## Document-Aware Expansion Experiment

The Mxbai experiment selected the top five documents from global hybrid candidates, fetched up to 50 vector-ranked chunks from each selected document, merged them with global candidates, and reranked with MiniLM. On v3 it scored 6/10 versus the unchanged Mxbai baseline's 8/10. It lost Q01 and Q02 and gained no new passage hits; therefore this configuration is rejected and was not tested on the holdout.

Expansion did recover the Q05 and Q08 gold chunks inside their expected documents at local vector ranks 24 and 47. The cross-encoder still placed both outside the final top five. The diagnostic script is `scripts/evaluate_document_expansion.py`; its result is `data/pilot_v1_parserfix_mxbai/results_v3_doc_expansion.json`. This suggests global candidate recall was improved, but the tested reranking method was not adequate to use those passages without degrading other questions.

## Expanded Holdout (2026-11-12)

A new 35-question set was created from 35 documents not previously used as benchmark targets: 17 canonical (10 Buffy HTML and 7 numbered Angel TXT) and 18 fanfiction. The prior 20 pilot documents remain in the corpus as distractors, giving 55 documents total. The frozen manifest is `data/pilot_v2_holdout/queries.json`; each relevant chunk ID was checked against its declared document and source class before evaluation.

All 55 files inventoried ready with no recovery warnings after using clean copies for known Angel control bytes. Parsing produced 15,553 chunks, 11,354 dialogue turns, and 82 parser warnings; FTS and citation validation passed. Mxbai indexed all 15,553 chunks with no failures.

Mxbai results: strict FTS Hit@5 0/35, semantic 21/35 (60%), and hybrid 22/35 (62.9%). The expected document appeared in the hybrid top five for 30/35 questions (15/17 canonical, 15/18 fanfiction), while the directly relevant passage appeared for 22/35 (12/17 canonical, 10/18 fanfiction). This indicates both document-level misses and passage-ranking misses remain. This set was used only for evaluation; no retrieval tuning was based on it.

Results are in `data/pilot_v2_holdout/results.json`. The index is isolated under `data/pilot_v2_holdout/storage/`; the active ScriptLab model and prior pilot databases were not changed.

### Canonical-Only Result

The 17 canonical questions were evaluated separately with the query source fixed to `canonical`; fanfiction remained present as distractor data but was ineligible for returned candidates. Mxbai scored FTS 0/17, semantic 10/17 (59%), and hybrid 12/17 (71%). By format, Buffy HTML scored 5/10 hybrid (expected document in top five for 8/10), while numbered Angel TXT scored 7/7 hybrid (expected document in top five for 7/7). This is a useful diagnostic split, not proof that Angel retrieval is solved; the subgroup is small and the questions differ in content.

The source-preserving canonical subset is `data/pilot_v2_holdout/queries_canonical_17.json`; results are in `data/pilot_v2_holdout/results_canonical_17.json`. Fanfiction target questions from the original 35 remain unchanged.

A source audit found H001's original wording could refer either to Willow checking on Xander immediately after the vampire attack or asking about Jesse after returning to the library. A separately versioned canonical v2 clarifies the library/Jesse question and labels the corresponding scene and dialogue chunk: `data/pilot_v2_holdout/queries_canonical_17_v2.json`. Its hybrid score remains 12/17, and H001's target document is still absent from the top five, so this is a document-recall miss rather than only a label issue. The original 35-query manifest and canonical v1 are unchanged; corrected-set results are `data/pilot_v2_holdout/results_canonical_17_v2.json`.

## Parser Boundary Fix

The demonstrated action leak was fixed incrementally through parser v16/chunker v14. A speaker block now ends before a clearly recognized action/camera paragraph after dialogue; grouped and named-character action paragraphs, plus common uppercase object inserts followed by physical actions, are not treated as spoken turns. Parentheticals remain supported. The ten-action audit below checks selected cases, not every line in the corpus; broader speaker/action accuracy remains unmeasured.

The fix was parsed and indexed in a separate database under `data/pilot_v1_parserfix/`, reusing the same 20 source copies. It contains 4,845 chunks and 4,291 dialogue turns; citation validation passed. The original pilot database, v1/v2 IDs, and results remain unchanged.

## HTML Layout Metadata

Parser v17/schema v5 preserves HTML layout cues as offset spans beside normalized text. In the 15 Buffy HTML files in the 55-document pilot, the new extraction recorded 4,746 centered spans, 250 right-aligned cues, 79 styled headings, and 47 act markers. The visible text and citation offsets remain unchanged.

Comparing fresh parser v16 and v17 parses of the same files, Buffy HTML dialogue turns changed from 6,273 to 6,203 and parser warnings from 22 to 1; Angel TXT was unchanged. The ten-action audit remains 10/10 for source, scene chunk, and citation-span preservation, and dialogue contamination changed from 2/10 to 0/10. The isolated parser v17 Mxbai index is complete (15,529/15,529); however, the same ten action questions remain 7/10 semantic and 6/10 hybrid. Improved structure has not yet improved retrieval on this query set. The layout spans and parser v17 index are isolated under `data/pilot_v2_holdout_parser17/`; the active ScriptLab model/index is unchanged.

Angel HTML alternatives were tested separately, without changing the agreed 55-document corpus scope. The updated extractor recognizes their Word-style `<p align="center" style="text-align:center">` dialogue blocks. Three copied episodes (`1x08.htm`, `1x14.htm`, `2x01.htm`) yielded 435/448/443 centered spans, 10/7/6 right-aligned cues, and three act markers each. After fresh import and parser v18/chunker v16 parsing, they produced 416/429/422 dialogue turns, zero parse warnings, and zero turns crossing act markers; citation validation passed. This probe is under `data/angel_html_layout_probe/`. Original Angel files and the 55-document pilot scope remain unchanged; retrieval was not evaluated on this three-file probe.

## Angel TXT Encoding Cleanup Probe

Angel `1x03.txt` contains 53 embedded four-byte `00 00 03 XX` sequences splitting English words and speaker cues (212 bytes total). A separate working copy under `data/angel_1x03_cleanup/` removed only those exact sequences and recorded before/after SHA-256 values in `sanitization_manifest.json`; the original hash is unchanged. The clean copy inventories as UTF-8 with no recovery warnings, then reparses to 20 `RACHEL` dialogue turns versus 18 in the prior automatic-recovery parse. Its normalized text is 52 characters shorter because the automatic path had left 52 trailing non-ASCII bytes after removing the control bytes. This is one-file evidence that the broader byte-pattern cleanup may improve speaker parsing; do not generalize it until other Angel TXT files are checked for the same exact sequence.

## Action Data Accuracy Audit

A ten-item canonical-only action audit checks each manually selected direction at three data layers: it remains in normalized source text, appears in a `scene` chunk, and is covered by that chunk's citation span. All ten passed all three checks on parser v14 and parser v16. The same action evidence was absent from dialogue-turn bodies on v16 (0/10 contamination), compared with 2/10 on v14. The two corrected cases were a named-character action after Willow's line and the uppercase `GYM BAG` insert; Diego's `(hands Cordy ice)` parenthetical is retained as a cue annotation, not counted as dialogue contamination.

The audit compares parser v14's 15,553 chunks/11,354 turns with parser v16's 15,536 chunks/11,352 turns in separate databases; both pass citation validation. Manifest and script: `data/pilot_v2_holdout/action_questions_10.json` and `scripts/evaluate_action_data.py`. Results: `data/pilot_v2_holdout/action_results_parser14.json` and `data/pilot_v2_holdout_parser16/action_results.json`.

A broader stratified manual sample reviewed 40 turns: five each from four Buffy HTML and four Angel TXT episodes, selected at distributed ordinal positions. All 40 detected labels matched the visible source cue line; this checks cue extraction, not character identity resolution or every line in the corpus. Twelve of 40 turns contained stage/action text inside the turn (3/20 Buffy HTML; 9/20 Angel TXT). This convenience sample is not a corpus-wide error-rate estimate. The sampled turn IDs and classifications are in `data/pilot_v2_holdout_parser16/speaker_action_audit_v1.json`.

The parser v16 corpus was separately reindexed with Mxbai: 15,536/15,536 chunks are ready, with no remaining failed jobs. The first pass had 16 transient localhost timeouts; the failed-only retry completed them. No existing index was changed.

Action-question retrieval is a separate score: both parser v14 and parser v16/Mxbai score semantic Hit@5 7/10, hybrid 6/10, and strict FTS 0/10 for these natural-language questions. Thus parser v16 improved the audited speaker/action data (2/10 to 0/10 contamination) without changing retrieval on this small action set. The directions are present and citable, but the current search still often fails to retrieve them. Exact-term FTS can find specific action words; the 0/10 result concerns this question wording and strict all-term matching.

The separately relabeled v3 benchmark uses the same v2 questions against parser v14/chunker v12 chunk IDs. Baseline results are in `data/pilot_v1_parserfix/results_v3.json`; the balanced-pool rerun is in `data/pilot_v1_parserfix/results_v3_balanced.json`.

A rank diagnostic showed Q07's gold passage at vector rank 7 and OR-FTS rank 28. Expanding both hybrid candidate pools to 50 surfaced it; using a narrower 50/50 pool with RRF `k=40` also retained Q10, for 7/10 total. All ten expected documents appear in the hybrid top five, but only 7/10 have a directly relevant passage there. Q04, Q05, and Q08 remain passage-level misses. Q04 is not in the top 50 vector candidates, Q05 is present but ranks well below five, and Q08 is absent from both the top 50 vector and top 250 lexical candidates. The parser correction improves evidence boundaries but does not solve these retrieval failures; the 8/10 target remains unmet.

## Reproduction

Run from the `ScriptLab` directory with Ollama available and the configured embedding model installed:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v1\scriptlab.yaml --queries data\pilot_v1\queries.json --output data\pilot_v1\results_after_hybrid_or.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v1\scriptlab.yaml --queries data\pilot_v1\queries_v2.json --output data\pilot_v1\results_v2.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v1_parserfix\scriptlab.yaml --queries data\pilot_v1_parserfix\queries_v3.json --output data\pilot_v1_parserfix\results_v3_balanced.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v1_parserfix_nomic\scriptlab.yaml --queries data\pilot_v1_parserfix\queries_v3.json --output data\pilot_v1_parserfix_nomic\results_v3.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v1_parserfix_mxbai\scriptlab.yaml --queries data\pilot_v1_parserfix\queries_v3.json --output data\pilot_v1_parserfix_mxbai\results_v3.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v1_parserfix_mxbai\scriptlab.yaml --queries data\pilot_v1_parserfix\queries_holdout1.json --output data\pilot_v1_parserfix_mxbai\results_holdout1.json
.\.venv\Scripts\python.exe scripts\evaluate_document_expansion.py --config data\pilot_v1_parserfix_mxbai\scriptlab.yaml --queries data\pilot_v1_parserfix\queries_v3.json --output data\pilot_v1_parserfix_mxbai\results_v3_doc_expansion.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v2_holdout\scriptlab.yaml --queries data\pilot_v2_holdout\queries.json --output data\pilot_v2_holdout\results.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v2_holdout\scriptlab.yaml --queries data\pilot_v2_holdout\queries_canonical_17.json --output data\pilot_v2_holdout\results_canonical_17.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v2_holdout\scriptlab.yaml --queries data\pilot_v2_holdout\queries_canonical_17_v2.json --output data\pilot_v2_holdout\results_canonical_17_v2.json
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py --config data\pilot_v1_parserfix\scriptlab.yaml --queries data\pilot_v1_parserfix\queries_holdout1.json --output data\pilot_v1_parserfix\results_holdout1_baseline.json
.\.venv\Scripts\python.exe -m pytest -q
```

The evaluator writes chunk IDs and filenames, not retrieved source excerpts. Pilot source copies, database, and result JSON files are ignored by Git.

## Priorities Before Further Analysis

1. Manually review a broader stratified script sample for remaining dialogue/action boundary errors.
2. Investigate the passage-ranking misses Q04, Q05, and Q08 without changing v1; use a separately versioned benchmark for any revised questions or labels.
3. Re-run the pilot gates after demonstrated parser or retrieval changes. Do not begin character voice analysis from the current dialogue-turn output until the broader review supports it.
