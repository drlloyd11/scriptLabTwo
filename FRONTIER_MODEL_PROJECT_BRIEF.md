# ScriptLab: Project Brief and Request for Technical Review

- **Status:** Working local retrieval prototype; relevance and performance require further work.
- **Snapshot:** November 13, 2026
**Code repository:** https://github.com/drlloyd11/scriptLabTwo

## Important repository caveat

The search improvements in this briefing followed commit `2563d93` ("Speed up
readiness checks and add search sessions") and are committed alongside this
document. They include hybrid readiness optimizations, SQLite schema v6,
per-document readiness counts, and opt-in search timing instrumentation. Push
that commit before expecting it to appear on GitHub. The rebuilt local
database has already been migrated to schema v6; older code expecting schema v5
cannot use it unchanged.

The actual scripts, fanfiction, SQLite databases, Chroma collections, local
configuration, and virtual environment are not in Git. This brief contains no
source excerpts or credentials. A reviewer who can only access the repository
cannot directly reproduce the full-corpus measurements without a separately
authorized local dataset and the relevant code.

## 1. What we are trying to do

ScriptLab is a Windows-first, local-first project for searching a corpus of
Buffy the Vampire Slayer and Angel scripts alongside fanfiction. The immediate
goal is **reliable, fast, source-grounded retrieval**, not merely generating
plausible answers.

The system should:

- Find passages that actually answer natural-language questions, even when the
  question and source use different wording.
- Support exact names, phrases, and dialogue through full-text search as well
  as meaning-based search.
- Keep canonical scripts and fanfiction separate unless the user explicitly
  requests both. Fanfiction must not be presented as canonical fact.
- Return traceable citations to the original normalized source spans.
- Provide enough context to distinguish a character's own dialogue, other
  speakers, narration, action, and chronology.
- Preserve original files and permit safe, resumable indexing and rebuilding.

Longer-term goals include evidence-backed character and relationship analysis,
continuity research, and generation of new scenes using retrieved evidence.
Those are design goals, **not claims that the current prototype implements
them**. We should not use it as an authoritative character-voice system until
retrieval and parser quality have been evaluated further.

## 2. Tools and environment

- Python project with a CLI; development is on Windows with Python 3.13 in a
  virtual environment. The package metadata currently declares Python 3.8+;
  that declaration is not by itself proof that every version has been tested.
- SQLite is the local authoritative store for documents, chunks, offsets,
  parsing records, vector-index metadata, and embedding jobs. SQLite FTS5
  supplies lexical search.
- ChromaDB stores persistent vector collections. Ollama provides local
  embeddings using `qwen3-embedding:0.6b` for the active full-corpus index.
- Beautiful Soup is used for HTML processing; YAML configures source locations,
  storage, model, and Ollama endpoint.
- `pytest` provides automated tests. The most recent full run reported
  **135 passed, 1 skipped**.
- The machine has approximately 16 GB of RAM and an NVIDIA laptop GPU.
  Search timings vary with system activity, caching, source filter, and query.
- The prototype itself runs locally. A hosted/frontier model is being asked for
  design advice; it is not the owner of the corpus or an implemented dependency.

## 3. Source material and corpus

The current rebuilt index contains **983 documents and 727,332 chunks**:

| Source class | Documents | Chunks |
|---|---:|---:|
| Canonical Buffy and Angel scripts | 212 | 16,630 |
| Fanfiction | 771 | 710,702 |
| **Total** | **983** | **727,332** |

The canonical set comprises 146 Buffy and 66 Angel documents in the tested
rebuild. The documents include HTML/HTM and TXT material with differing script
layouts; fanfiction is prose and may contain alternate interpretations or
continuities. ScriptLab currently supports direct-child TXT, HTML, and HTM
sources, not every format proposed in the broader specification.

The chunk inventory contains 695,757 prose passages, 23,939 scenes, and 7,636
dialogue exchanges. Fanfiction accounts for most chunks, creating a heavily
imbalanced search space. Scene and dialogue chunks can overlap; some prose or
dialogue chunks are extremely short. Consequently, a high-ranking chunk may
contain a character name without enough context to answer a question.

Original files remain unchanged. Extracted text is normalized to LF line
endings. HTML layout cues are stored separately as offset-based metadata; they
do not insert markers into the normalized source. TXT recovery is recorded
with warnings. Composite chunks may consist of several source spans, joined for
display, and must retain the individual offsets needed to reconstruct them.

Current code defaults to parser `scriptlab-parser-v18` and chunker
`scriptlab-chunker-v16`; all chunks in the rebuilt index report chunker v16.
The chunker label alone does not establish the parser version for every
previously parsed document. Recent parsing improvements increased confidence
in particular script layouts, but manual review remains incomplete.

## 4. Architecture: implemented versus planned

### Implemented retrieval path

1. **Inventory and import:** Scan configured source folders, hash inputs,
   extract and normalize text, and store documents with source class and
   metadata. Reimport of unchanged files is idempotent; changed inputs are
   reported as conflicts rather than silently replacing stored source.
2. **Parse and chunk:** Deterministic parsing identifies scenes, dialogue,
   prose passages, and warnings. Chunks preserve source spans and stable IDs.
3. **Lexical indexing:** SQLite FTS5 indexes chunk text. FTS synchronization is
   maintained by database triggers.
4. **Vector indexing:** Ollama embeds chunk text; Chroma stores the vectors.
   SQLite tracks the index version and per-chunk jobs. Indexing is resumable;
   failed jobs can be retried, and a model change uses a separate staged
   collection rather than mixing incompatible vectors.
5. **Retrieval:** A user selects FTS, semantic, or hybrid mode and a source
   class. Chroma candidates are resolved through SQLite, where source/work
   filters and document readiness are checked. Hybrid combines lexical and
   vector ranks using reciprocal rank fusion.
6. **Output:** The CLI returns ranked chunks with metadata and source spans.
   It does **not** currently synthesize an answer from those chunks. A
   `search-session` keeps searches in one process; optional `--timings` reports
   stage durations and candidate counts.

SQLite is the authority for source text and readiness; Chroma is a separate
materialized vector index. SQLite job readiness does **not** independently
prove that every expected vector is still present in Chroma. Do not weaken
source isolation, citation integrity, or incomplete-document exclusion as a
performance shortcut.

### Planned, not implemented by this retrieval prototype

The broader design proposes evidence-backed claims, alias resolution,
relationship and chronology models, token-budgeted context packages, a
localhost API, optional MCP integration, approved hosted-model adapters, and
writing/review workflows. These should not be treated as existing features.

## 5. Code and current search behavior

Principal modules in the repository:

| Module | Responsibility |
|---|---|
| `src/scriptlab/config.py` | YAML configuration and path validation |
| `src/scriptlab/corpus.py` | Source scanning, hashing, decoding, HTML/TXT extraction, and layout metadata |
| `src/scriptlab/parser.py` | Conservative scene, dialogue, and prose parsing; span-based chunk creation |
| `src/scriptlab/database.py` | SQLite schema/migrations, imports, FTS, readiness, search, and citation hydration |
| `src/scriptlab/vectors.py` | Ollama embedding, resumable index jobs, Chroma adapter, and semantic retrieval |
| `src/scriptlab/cli.py` | Commands, mode/source selection, hybrid fusion, session, and user-facing output |
| `src/scriptlab/timing.py` | Opt-in per-search timing and counters |
| `scripts/evaluate_pilot.py` | Frozen-query retrieval evaluation |
| `scripts/evaluate_action_data.py` and `scripts/evaluate_document_expansion.py` | Separate parser/action and document-expansion experiments |

The current search behavior is important to the problem:

- Query terms are tokenized with a word-character regex. Standalone FTS
  requires **all** terms; hybrid's lexical branch uses **OR** matching.
  There is no demonstrated stop-word removal, entity-aware query rewriting,
  or temporal query understanding in this path.
- For a hybrid request for five results, the CLI requests up to **50 FTS
  results** and **50 validated vector results**. The vector layer initially
  asks Chroma for up to **500 raw candidates** and can increase that request
  if too few valid candidates remain.
- Each branch ranks its candidates independently. Fusion sums
  `1 / (40 + rank)` across branches and returns the top five. A chunk appearing
  in both branches receives contributions from both. This is the current code's
  constant, not the `60` shown as an example in the proposed design.
- Candidate validation preserves source/work filters and excludes incomplete
  documents. Citation formatting is currently performed for candidates before
  final fusion, not only for the final five.
- Search results are evidence passages, not verified claims or a generated
  response. Passing indexing or citation checks does not prove relevance.

The latest local schema uses transactionally maintained document chunk counts
and ready-job counts per index version. Searches compare these counts through
indexed SQLite lookups rather than rescanning the entire corpus for every
candidate. Tests cover failed and missing jobs, incomplete nonmatching chunks,
reparsing/reindexing, migration, rollback, and candidate ordering.

## 6. Progress and evidence

### Index and integrity

- The isolated rebuild completed: **727,332 / 727,332 ready jobs**, 983 / 983
  documents ready, and no pending or failed jobs in the measured active index.
- Full FTS and citation validation previously passed for the rebuilt corpus.
  Validation confirms indexing/span integrity, not retrieval relevance.
- Schema v6 was tested on a full database copy; its counts were compared
  against the underlying chunks and jobs before migration of the working
  database. A pre-migration SQLite backup was retained locally.
- Source filtering, citation presence, ordered candidate resolution, and
  incomplete-document behavior have automated coverage. The latest full test
  suite passed as noted above.

### Earlier relevance work

The tracked [pilot report](PILOT_REPORT.md) documents separate small corpora and
several embedding/ranking experiments. These are **not full-rebuild scores**:

- On a separately audited ten-question pilot, Qwen hybrid reached **7/10**
  passage Hit@5 with the balanced 50/50 candidate configuration.
- A separate Mxbai experiment reached **8/10** on that pilot, but only **5/10**
  on its ten-question holdout.
- An expanded, isolated 55-document Mxbai experiment reached **22/35
  passage Hit@5** on a 35-question holdout. The expected document appeared in
  the top five for 30/35, illustrating a difference between document retrieval
  and retrieval of the relevant passage.
- A tested cross-encoder reranker and a document-expansion experiment degraded
  their respective pilot baselines and were not adopted.

These are small, versioned experiments using different corpora, indexes, and
in some cases models. They do not establish current full-corpus relevance.
Existing pilot chunk IDs depend on source paths and chunker versions and do not
directly match the rebuilt index's IDs; labels must be carefully remapped and
verified before reuse. Preserve held-out questions for evaluation rather than
tuning directly against them.

### Timing and candidate-pool observations

- The first Chroma operation in a process can be expensive. In one canonical
  hybrid search the first run took **39.3 s**, including **15.3 s** in
  collection count; the repeat took **15.6 s**, with an **8.4 s** SQLite status
  spike. A different canonical hybrid run later took about **3.2 s**.
- A fanfiction hybrid session took **93.5 s** initially and **33.4 s** on
  repeat in one instrumented run. These figures varied substantially between
  runs; they are observations, not guaranteed service levels or a controlled
  before/after comparison.
- In a 20-search vector-stage run, SQLite status exceeded one second 7 times,
  but stayed below 0.5 seconds in the final eight searches. Chroma candidate
  lookup and startup also varied. A later idle system snapshot does not explain
  what happened during earlier spikes.
- We compared raw Chroma pools of 50, 100, 250, and 500 on eight sample
  query/source combinations. All produced the same ordered hybrid top five,
  with citations reconstructing correctly. In a separate small
  counterbalanced fanfiction retest, warm vector-stage medians were **3.62 s**
  for 50 versus **3.72 s** for 500. These samples do not establish that a
  smaller pool preserves relevance generally, or that it reliably improves
  speed; production remains at 500.
- `--timings` prints wall-clock stage durations and counters to stderr.
  Substage durations overlap their enclosing totals. Retrieval total excludes
  configuration loading and printing results.

## 7. Roadblocks and the critical relevance failure

### Birth-date example

The user asked: **"when was buffy born"**, with canonical source filtering.
The hybrid search took about **29 seconds** and none of the five returned
passages answered the question. The top result concerned a different
character's birth; other results were unrelated fragments.

The gravestone in Buffy episode 5x22 records the year **1981**. We verified
that this year exists in the stored normalized document and is present in two
FTS-indexed chunks. Nevertheless, chunks containing that year were absent from
both the 50 returned lexical candidates and the 50 returned vector candidates
for this question. Thus, for this example, the evidence was present and
indexed, but candidate retrieval failed to surface it. This observation alone
does not identify the exact cause.

Birthday episodes also provide useful distinct, time-specific examples:
season 2 episode 13 discusses her 17th birthday; season 3 episode 12 her 18th;
and season 4 episode 12 her 19th. They should be checked as separately labeled
source passages when constructing a benchmark. A question asking "how old is
Buffy?" is temporally ambiguous without a season or event; a system should
not collapse multiple ages into a single timeless fact. The gravestone alone
establishes a year, not an exact month and day.

### Open questions

1. **Candidate recall:** Why do lexical and vector retrieval both miss
   known-relevant source chunks? Measure their actual ranks and scores, not
   only the final fused results. Inspect whether embedding input length,
   context loss, query wording, metadata filtering, or another factor matters.
2. **Lexical noise:** Hybrid OR matching admits common words and isolated
   character labels. Determine whether better query processing, phrase
   handling, stop-word treatment, or another approach improves relevance
   without sacrificing exact-match retrieval.
3. **Chunk quality and context:** Short fragments, whole scenes, and
   overlapping dialogue chunks may produce redundant or misleading rankings.
   Test whether context expansion, chunk sizing, or grouping improves the
   evidence returned while retaining exact citations.
4. **Ranking and fusion:** Determine whether failures arise before fusion,
   in branch ranking, or in fusion. Do not assume a reranker can recover a
   passage missing from both candidate sets.
5. **Latency:** Separate cold-process startup from warm query costs; measure
   Chroma source-filter behavior, SQLite status spikes, FTS time, and citation
   hydration. Do not attribute a spike to memory, disk, or filter design
   without measurement.
6. **Evaluation:** Build a representative, manually labeled, source-specific
   test set for the rebuilt corpus. Include direct fact lookup, paraphrase,
   names, age/chronology, exact quotations, fanfiction, and adversarial source
   isolation. Measure passage and document recall, branch candidate recall,
   Hit@5, citation correctness, and cold/warm latency separately.

## Request to the reviewing model

Please assess the actual implemented retrieval design, not just the intended
future architecture. Propose a dependency-ordered investigation and a minimal,
testable set of changes that improves **passage relevance first**, then latency.
For each recommendation:

- State the observed evidence, the hypothesis, and how to falsify it.
- Specify what to measure on FTS, vector, and fused candidates independently.
- Identify possible regressions, especially source leakage, false certainty
  about chronology, incomplete-index results, citation errors, and approximate
  vector-ranking changes.
- Preserve a clean held-out evaluation and compare against the existing
  implementation before proposing a new model, reranker, index, or query router.
- Do not assume that faster indexing, a larger candidate pool, successful
  citations, or a fluent generated response demonstrates retrieval quality.

Relevant repository documentation: [README](README.md),
[pilot report](PILOT_REPORT.md), and
[corpus ingestion and indexing specification](SCRIPTLAB_CORPUS_INGESTION_AND_INDEXING_SPEC_v1.3.md).
