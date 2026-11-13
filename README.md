# ScriptLab

The next iteration of scriptLabTwo.

ScriptLab is a local prototype for searching a fictional-universe corpus while
keeping canonical scripts and fanfiction distinct and preserving evidence for
later citations.

## Set up on Windows

From this directory, create and activate a virtual environment, then install
the project and its test dependencies:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,vectors]"
```

Create a sample configuration. This writes `scriptlab.yaml` only; it does not
create or modify source directories:

```powershell
scriptlab init
```

The sample configuration lists the Buffy and Angel shooting-script folders as
canonical sources and `kitten_fanfics` as the fanfiction source. Edit these
paths if your folders are elsewhere, then check the CLI:

```powershell
scriptlab --help
scriptlab inventory
scriptlab ingest
scriptlab status
scriptlab parse
scriptlab preview "C:\Users\reven\kitten\Buffy shooting scripts\1x01.htm"
scriptlab search "Giles argues with Buffy" --mode fts --source canonical
scriptlab search "someone conceals a secret" --mode semantic --source canonical
scriptlab search "Giles challenges Buffy" --mode hybrid --source both
scriptlab search-session --mode semantic --source canonical
scriptlab search "Watcher" --mode semantic --fallback-to-fts
scriptlab validate
scriptlab index --batch-size 16 --limit 100
python -m pytest
```

Relative paths in the configuration are resolved from the configuration file's
directory. A different configuration can be selected with the global option
before the command, for example:

```powershell
scriptlab --config .\scriptlab.yaml status
```

`inventory` reads only direct-child `.txt`, `.html`, and `.htm` files and writes
metadata, SHA-256 hashes, chosen encodings, and extraction outcomes to
`data\inventory_report.json`. It does not store extracted source text.
Unsupported file types are listed without being opened. Nested directories,
symlinks, and junctions cause preflight to fail before any source file is read.
`preview` prints extracted text for one direct-child source file; use
`--max-chars` to limit the displayed excerpt. TXT and HTML decoding is strict:
UTF-8 (including BOM) is tried first, then Windows-1252. As authorized by the
owner, TXT recovery removes suspicious control/replacement characters and,
when strict Windows-1252 decoding fails, only its undefined bytes. Newlines,
tabs, form feeds, punctuation, and all other decoded text remain unchanged.
Original source files are never modified. Every removal is counted by
character/byte code in the report and shown as a recovery warning by the CLI.
Recovered files count as ready but are also counted separately as recovered;
the encoding fallback remains uncertain. TXT with no usable remaining text
and HTML with decoding/control-character issues still fail explicitly.
HTML scripts, styles, navigation, and
hidden elements are omitted; `<br>`, paragraph boundaries, and `<pre>` content
are preserved as text layout. HTML imports also retain offset-based layout
metadata for centered blocks, right-aligned cues, styled headings, and act-end
markers. This metadata lives alongside normalized text and does not insert
markers into the source text or change citation offsets.
`ingest` stores complete extracted text with LF line endings in the
ScriptLab-owned `data\scriptlab.sqlite3` database. Stable IDs are derived from
normalized absolute source paths. Repeated imports skip unchanged files;
changed bytes or a source-class change at an existing path are reported as
conflicts and never overwrite stored text. Extraction recovery warnings are
retained with each document. `status` displays saved work/document counts and
per-file import issues.
On first use, the database is upgraded from schema v1 to v6 in transactions;
existing documents are preserved, and committed chunks are indexed in FTS5.
`parse` populates scenes, dialogue turns,
chunks, spans, and review warnings for all stored documents; pass a document ID
to limit it to one. `inspect <document-id>` shows headings, original speaker
labels, unresolved attribution, source offsets, span snippets, and warnings.
Fanfiction speakers are not inferred. Composite chunk text uses an explicit
`[...]` separator, but citations always resolve to the ordered source spans,
not the bounding range.
`search` uses FTS5 with AND matching for query terms. It defaults to canonical
sources; fanfiction or both source classes must be requested explicitly. Results
include source class, chunk type, exact source spans, and available speaker/scene
labels. `--mode semantic` searches Chroma only; `--mode hybrid` expands lexical
candidates with OR term matching and combines them with vector ranks using
reciprocal-rank fusion. Standalone FTS continues to require all query terms.
Both vector modes require a complete active index and query embeddings from the
configured Ollama model. Vector IDs
are resolved through SQLite, where source class, work, and full-document
readiness are verified; incomplete documents are excluded and coverage is
reported. SQLite maintains per-document chunk and ready-job counts transactionally,
so searches can check readiness without rescanning every chunk. These counts
track SQLite jobs, not independent verification of Chroma contents.
Use ScriptLab to modify its database; direct external writes are not supported.
Vector failures are errors by default. `--fallback-to-fts` explicitly
allows semantic or hybrid requests to run FTS instead, and output reports both
the requested and actual modes. `validate` checks FTS consistency and
reconstructs chunks from their ordered spans. `index` resumes Ollama embeddings
into a persistent Chroma collection and reports per-document vector readiness.
Use `--limit` to process a bounded number of new vectors per run and repeat the
command to resume. A `--retry-failed` run targets only failed jobs. A model
change requires `--rebuild`, which stages a separate collection and keeps the
active index until the replacement verifies.

`search-session` accepts one query per line in the same process, allowing
successive vector searches to reuse Chroma after its initial load. It defaults
to semantic search and accepts the same search options as `search`. Enter
`/quit` to exit; EOF also ends the session. Individual search errors are
reported and the session continues. End and restart the session before
reindexing or otherwise modifying the corpus or vector index.

Add `--timings` to `search` or `search-session` to report per-query retrieval
timings and candidate counts to stderr without changing search results. The
retrieval total excludes configuration loading and printing results; nested
stages overlap and should not be added together. Counts are totals across
candidate-query retries. Example:

```powershell
scriptlab --config data\rebuild_2026-11-12\scriptlab.yaml search-session --mode hybrid --source fanfiction --limit 5 --timings
```

If extraction failures occur, the report is still written and `inventory`
returns exit code 1. Unsafe folder layouts stop preflight with exit code 2
before source files are read or a report is written.

## Data safety

ScriptLab's configured storage is `.\data`, resolved relative to this project
directory. It is separate from the existing forum archive databases, including
`forum_chroma`, and ScriptLab must never open, modify, or delete those stores.
The Step 6 vector index must live in ScriptLab's own storage.

Building or rebuilding vectors must not destroy the currently usable index.
Create and verify a separate staging collection first; only make it active
after its expected IDs, counts, and source links pass validation. Keep the
previous active collection until the replacement is verified and working.
Never use a destructive reset or delete an existing collection as part of
routine indexing or rebuilding.

## Current implementation scope

Steps 1 through 7 provide source inventory, SQLite corpus persistence,
conservative scene/dialogue/prose parsing, source-linked chunks, warnings,
document inspection, FTS5 keyword search, resumable local vector indexing, and
semantic/hybrid retrieval. Vectors are built only when `scriptlab index` is run.

The committed TXT and HTML fixtures under `tests/fixtures/` contain synthetic
text for future tests; they are not real scripts or fanfiction.
