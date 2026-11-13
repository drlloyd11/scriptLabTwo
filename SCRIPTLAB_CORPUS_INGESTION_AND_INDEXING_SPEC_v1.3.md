# ScriptLab — Corpus Ingestion and Indexing

**Claude Code implementation specification · Phase 1 foundation · Version 1.3 · 2026-11-08**

**Status:** Revised for the eleven prior review findings and the owner-confirmed source layout (v1.3); first implementation work package. This document defines a focused initial build, not the entire ScriptLab application.

## 0. Mission and governing decisions

ScriptLab exists to **make an LLM understand a fictional universe well enough to write convincing original material within it**. The initial technical prerequisite is a reliable, source-linked corpus containing scenes, dialogue, speakers, and searchable passages.

This first implementation **does not** attempt character psychology, narrative theory, script generation, a premium LLM gateway, or MCP. It creates the evidence foundation they will use.

### Fixed project assumptions

- **Environment:** Windows 11; Ollama local service; NVIDIA RTX 4060 Laptop GPU (8 GB VRAM); 16 GB RAM; Python 3.11 recommended.
- **Corpus:** Hundreds of documents overall. Start with a pilot of **10 canonical scripts and 10 fan-fiction stories**, preferably from one fictional universe.
- **File types and layout:** Only `.txt`, `.html`, and `.htm` are supported in v1. Each registered source root is a flat ordinary Windows directory: supported files are directly inside the root, with no nested directories, symlinks, or junctions. PDF, DOCX, EPUB, OCR, ZIP archives, recursive traversal, and remote sources are out of scope.
- **Source types:** Imported content has exactly two stored source classes: `canonical` and `fanfiction`. A user-facing script root is registered as `canonical`; a fanfiction root is registered as `fanfiction`. The operator registers **separate, clearly labeled, non-overlapping source roots** with an explicit class once per root; the importer inherits classification from that root, never from per-file guesses or filenames. No source root contains mixed classes. Classification remains fixed after import. Generated content is stored elsewhere and never silently imported as source evidence.
- **Corpus behavior:** Imports are expected to remain stable. Routine source updates, deletion/purge, complex version promotion, and granular rights policies remain out of scope. An **explicit, rare operator conflict-resolution workflow** is supported; it preserves the old imported record and never silently overwrites it.
- **Originals:** Never alter or delete source files. Store paths and content hashes; normalized text is a durable local derivative.
- **Local-use policy (owner decision):** For this personal v1 prototype, the project owner has chosen unrestricted **in-app local** storage, indexing, analysis, and retrieval of the corpus classified as canonical or fan fiction. This supersedes the earlier hardening addendum's proposed deny-by-default local-use workflow **only for this pilot**. It is not a license determination or proof of legal authority. **No external transmission, hosted provider integration, public service, or automatic export of source excerpts is implemented in this phase.** Revisit provider-specific rights and egress controls before premium-model functionality is added.
- **Model use:** No LLM needed for deterministic ingestion. Use `qwen3-embedding:0.6b` via Ollama for vectors; **do not call `qwen3:8b` for character analysis yet**.
- **Stores:** SQLite is authoritative for identities, document text, structured records, job state, and provenance. SQLite FTS5 enables lexical search. Chroma provides a rebuildable vector index.
- **Priority:** Correct extraction, grounding, and retrieval over import speed.

### Relationship to earlier project documents

- **Controlling product direction:** `ScriptLab_Revised_Project_Vision_v0.2.md` — world, character, and narrative models are the ultimate goals; this phase builds their evidence layer.
- **Prior architectural reference:** `ScriptLab_Architecture_Blueprint_v0.1.docx` — use for overall local RAG concept.
- **Prior design-hardening reference:** `ScriptLab_Design_Hardening_Addendum_v0.3.md` — retain its sound ideas about source references, integrity, and repeatable indexing, but **do not implement** deferred per-document permissions, exact-payload approval, lifecycle version promotion, secure purge, or elaborate retention machinery now.
- **Owner-approved scope reconciliation:** The owner has explicitly selected the narrow unrestricted-local-use assumption and stable canonical/fan-fiction corpus for the personal prototype. Therefore, the hardening addendum's granular permission engine, per-send payload approval, and forensic purge are deferred, **not silently implemented or regarded as legally waived**. The structural integrity, provenance, crash recovery, and evaluation requirements remain. Any later external transmission requires a new design/owner review. If other conflicts appear, identify them before expanding scope.

## 1. Definition of done

A user can point the CLI at a folder containing scripts and/or fan fiction and obtain:

1. An inventory of the configured, disjoint canonical/script and fan-fiction roots containing only TXT/HTML files directly in each flat root, with inherited root classification and SHA-256 hashes.
2. Normalized text in SQLite, with document metadata, explicit logical-work identity, and stable source references.
3. For scripts, identified scenes, dialogue turns, speaker labels, and stage directions wherever deterministically supported.
4. For fan fiction, coherent prose passages and explicitly attributed dialogue where reliable; ambiguous attribution remains unknown.
5. Searchable SQLite FTS5 passages and Chroma vector embeddings for *dialogue exchanges* and *scene/passages*.
6. A work-scoped hybrid search CLI that returns the passage, source file, source type, source offsets, and speaker/scene metadata where available.
7. Re-runnable imports and embeddings without duplicate records; resumable work after interruption.
8. Automated tests and a compact human review report for parsing and retrieval quality.

**Do not claim completion unless tests pass on realistic pilot files.**

## 2. Deliverables and repository layout

Create a Python package with a console entry point `scriptlab`. Proposed layout:

```text
ScriptLab/
  pyproject.toml
  README.md
  .gitignore
  config/
    settings.example.yaml
  src/scriptlab/
    __init__.py
    cli.py
    config.py
    db.py
    schema.sql
    models.py
    inventory.py
    text_extract.py
    normalize.py
    parse_script.py
    parse_fanfiction.py
    segment.py
    ingest.py
    fts.py
    embeddings.py
    vector_store.py
    search.py
    validation.py
  tests/
    fixtures/
      sample_script.txt
      sample_fanfiction.txt
      sample_script.html
    test_schema.py
    test_extraction.py
    test_script_parser.py
    test_fanfiction_parser.py
    test_ingestion.py
    test_indexing.py
    test_search.py
    test_works.py
    test_visibility.py
    test_conflicts.py
    test_path_identity.py
    test_work_hierarchy.py
    test_uniqueness.py
    test_search_modes.py
    test_source_roots.py
  data/                   # gitignored: databases, indexes, extracts
  corpus/                 # optional; real corpus need not be copied here
```

Use `pyproject.toml` and a venv. Keep the first release CLI-only. Streamlit, APIs, and MCP belong to later phases.

## 3. CLI and work-identity contracts

`work_id` is a **stable identifier for a logical creative work**, not a filename and not a raw hash. A television episode is normally one work, even if represented by multiple TXT or HTML files; a fan-fiction story is normally another work. A television **series** can be a parent work of individual episodes. Source files always have their own `document_id`.

### 3.1 Work creation and validation

- `scriptlab work create --kind series --title "Example Series"` creates a UUIDv4 work ID and prints it. `--kind` accepts `series`, `episode`, `fanfic`, `other`. The CLI creates IDs; callers cannot supply arbitrary IDs through `work create`.
- `scriptlab work create --kind episode --title "Episode 1" --parent-work-id <series-id>` creates an episode under a series. `fanfic` may have a series parent to denote universe association, but is still classified as `fanfiction` at import.
- `scriptlab work list` and `scriptlab work show <work-id>` let the operator inspect assignments.
- Reject blank/malformed/nonexistent IDs, parent cycles, invalid parent kinds, and inconsistent source classification. **Only leaf works** (`episode`, `fanfic`, `other`) carry an assigned `source_class`. A `series` parent always has `source_class=NULL` and may contain canonical and fanfiction descendants. A `fanfic` leaf is always `fanfiction`; an `episode` leaf is always `canonical`; an `other` leaf accepts either class on first import and locks it. Enforce the rules with SQLite CHECK constraints/triggers and service validation. Full matrix and rejection tests appear in §3.3.
- `--work-id` attaches all files processed by **that invocation** to the same existing work; show a warning/confirmation if more than one distinct file is being assigned to an episode or fanfic work. This is appropriate for different editions of the same work, but **not** for an entire season of different episodes.
- For bulk folders with many works, `--work-map <CSV>` maps **normalized root-relative paths** to existing work IDs. Required columns: `relative_path,work_id`; every supported file must have an entry, and mappings must be validated before writing. Multiple rows may intentionally share a work ID (different editions). Duplicate or ambiguous path mappings are rejected using the Windows path-normalization contract in §3.4.
- `--create-works-per-file` is an explicit convenient fallback: create one new work per source file with a generated ID and filename-derived *provisional* title, recording that identity has not been manually verified. This never merges files based on filenames, even if they appear identical as works.
- For `ingest`, **exactly one** of `--work-id`, `--work-map`, or `--create-works-per-file` is required. For `inventory`, the same options may be supplied for a read-only assignment preview (or omitted for file-only inventory).
- Every `inventory` or `ingest` invocation selects **exactly one registered source root** with `--root-id` or `--root`. An optional `--all-roots` inventory command can aggregate read-only results by root, but bulk ingestion happens per root so each mapping CSV is unambiguous. CLI `--source` / `--source-class` must **not** be accepted on these operations: source classification comes exclusively from the registered root. A direct file import must be contained in a registered root, and uses that root's identity.

### 3.2 Commands

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Register two independent, non-overlapping flat roots and retain the printed root IDs.
# Script files use the canonical source class.
scriptlab source-root add "D:\ScriptSources\Scripts" --class canonical
scriptlab source-root add "D:\ScriptSources\Fanfiction" --class fanfiction
scriptlab source-root list

# These are read-only. Classification is inferred from the registered root.
scriptlab inventory --root "D:\ScriptSources\Scripts"
scriptlab inventory --root "D:\ScriptSources\Fanfiction"
scriptlab inventory --all-roots

# Create logical works as necessary (printed IDs are used in subsequent commands)
scriptlab work create --kind series --title "Example Series"
scriptlab work create --kind episode --title "Example Episode" --parent-work-id <series-id>

# Import one file; its containing registered root determines source class
scriptlab ingest --root "D:\ScriptSources\Scripts" --file "episode01.txt" --work-id <episode-id>

# Import a flat root with explicit work assignments, or one provisional work per file
scriptlab ingest --root "D:\ScriptSources\Scripts" --work-map "D:\canon_work_map.csv"
scriptlab ingest --root "D:\ScriptSources\Fanfiction" --create-works-per-file

# Index from committed SQLite chunks
scriptlab index --collection all

# Work-scoped retrieval is unchanged
scriptlab search "sarcastic response to danger" --work-id <episode-id> --source canonical --limit 5
scriptlab search "betrayal" --work-id <series-id> --include-descendants --mode hybrid
scriptlab search "betrayal" --work-id <series-id> --include-descendants --mode fts
scriptlab search "betrayal" --work-id <series-id> --include-descendants --mode semantic
scriptlab search "betrayal" --work-id <series-id> --include-descendants --mode hybrid --fallback-to-fts

scriptlab status
scriptlab validate
scriptlab conflicts list
```

**Scope:** `source-root add` takes an explicit class from the operator and validates the directory and root overlap. `inventory` / `ingest` never accept or infer a class from a file name. `search --source` remains a **retrieval filter**, not an import classification switch. `--root` identifies an already registered path; `--root-id` is an equivalent alternative. `--file` is a root-relative filename directly inside the flat root; paths containing directory components are rejected. `--all-roots` is read-only in v1 and runs the same flat-root checks independently for each registered root. Ingestion always selects exactly one root.

The `--work-id` filter selects that exact work by default. `--include-descendants` expands a series to child works using a validated recursive work hierarchy. **Do not search across unrelated works by default:** `search` requires `--work-id` (with optional descendants) unless an explicit `--all-works` flag is provided. `--all-works` must be visibly labeled as a cross-work search. Programmatic callers follow the same rule.

Additional behaviors:

- `inventory` and `ingest --dry-run` make **no DB or Chroma changes**; dry run validates work mapping and prints intended assignments.
- Missing paths, unsupported formats, unknown work IDs, invalid CSV mappings, missing Ollama, or corrupt Chroma produce nonzero exit codes at the appropriate stage.
- Ingestion works while Ollama is offline; vector indexing does not. `--root` / `--root-id` is mandatory for ingestion, its registered class is authoritative, and only direct child files of the flat root are considered.
- `status` reports works, documents, scenes, turns, chunks, FTS-ready documents, hybrid-ready documents, pending/failed indexing jobs, and configuration.
- `search` returns stable source citation IDs (`document_id:chunk_id`) and actual `work_id`, title, file, offsets, and readiness mode.

### 3.3 Work hierarchy and source-class invariant (normative)

| `works.kind` | May be parent of | May itself be imported into? | `works.source_class` |
|---|---|---|---|
| `series` | `episode`, `fanfic`, `other` | **No** | Always `NULL` |
| `episode` | None | Yes, `canonical` only | Fixed `canonical` |
| `fanfic` | None | Yes, `fanfiction` only | Fixed `fanfiction` |
| `other` | None | Yes, first import `canonical` or `fanfiction` | Assigned on first successful import and never changes |

Exactly one parent level is supported in v1: a leaf may have **no parent** or a `series` parent; a `series` has no parent. No leaf has children; no series-within-series nesting. `--include-descendants` for a series means its direct leaf children; asking it for a leaf is rejected. `works.source_class` is NULL only for `series` and for unassigned `other` until first successful import. For `episode`/`fanfic`, set it upon creation, not through the first imported file. A failed import must not lock an unassigned `other` work. Direct imports to series IDs are errors. Validate these with SQL constraints/triggers (including attempted direct SQL INSERT/UPDATE), not only CLI checks.

### 3.4 Windows path identity and work-map contract (normative)

Treat an **import root** as a *pre-registered, flat directory* identity. Registration records its immutable `source_class` (`canonical` for scripts or `fanfiction`) and refuses overlapping root directories, even when different classes are requested. A single-file import does **not** implicitly register its parent folder; it must select a registered root and pass a root-relative `--file` naming a direct child. Store a `source_roots` row with a stable `root_id` UUIDv4, `canonical_root_path`, `root_key`, and `source_class NOT NULL CHECK (source_class IN ('canonical','fanfiction'))`. Once registered, changing the class is prohibited; the operator may choose a new, non-overlapping root for another class. `root_key` is the canonical absolute Windows path after normalizing separators, removing redundant `.`/`..`, applying Unicode NFC normalization and Unicode `casefold()`; reject UNC/device paths if they cannot be normalized safely. Repeated registration with the same `root_key` reuses the same `root_id`. Symlinks and junctions are not part of the supported corpus layout: do not follow them; report them and fail inventory/ingestion before writing any records. Nested directories are unsupported; inventory and ingestion must report them and fail before writing any records. An import root must exist; a missing root is an error.

For a discovered file, reject reparse points (including file symlinks, directory symlinks, and junctions) before opening them. The file must be a direct child of the resolved root. Compute its root-relative filename with `/` separators and NFC normalization. Reject absolute paths, drive letters, directory components, `..`, empty names, trailing separators, and escaped locations in CSV mapping keys. Define `relative_path_key = NFC(filename).casefold()` for lookup and uniqueness on Windows. Case-conflicting filenames in one root are an explicit collision/error rather than silently selecting one. Store original display spelling separately. Do **not** use Unicode folding or case rules intended for a different platform if a future Linux implementation is added; Windows is the v1 contract.

`source_path_identity = SHA256(UTF8(root_id + "\n" + relative_path_key))`; persist both columns and the digest. `UNIQUE(root_id, relative_path_key)` exists in the `source_slots` table. This identity persists across changed content at the same location. CSV `relative_path` is a filename directly under the registered root, normalized by the same rules, and must not contain a directory component or absolute path. Validate CSV duplicates after normalization, unknown/missing files, and conflicting work IDs **before any write**. `--work-map` with a single-file import still uses the import root, not the map file's directory.

**Identical bytes at a different path:** These are distinct imported documents and receive separate `document_id` and `source_path_identity` records; the content hash can be shared and recorded as `possible_duplicate`, but no automatic deduplication or work merging occurs. The operator may assign both representations to the same *existing* work using `--work-id` / `--work-map`. Reimporting identical bytes at the **same path identity** and same work/source classification is idempotent; a mismatched work assignment is a conflict, not a silent move. Separate roots containing the same absolute physical path after resolution are rejected as overlapping imports unless explicitly registered as the same root.

### 3.5 Separate source-root classification (v1.3 normative)

- The operator explicitly registers a **script root** (`source_class=canonical`) and a **fanfiction root** (`source_class=fanfiction`), or additional flat roots with one declared class each. Every root has exactly one class, and a root is not allowed to contain files of the other class by project convention. ScriptLab cannot infer a file's *true* rights or literary status from its text; it trusts the operator's directory assignment.
- The root name is not itself a classification algorithm: use the `source_roots.source_class` stored at `source-root add` time. It is displayed by `source-root list`, `inventory`, `ingest --dry-run`, and `status`. A discovered file receives `documents.source_class = source_slots.source_class = source_roots.source_class` through validated relationships; the user cannot override the class per file.
- `source_roots` must reject (a) the same path registered under conflicting classes, (b) parent/child overlap (even if both have the same class), and (c) alternate paths that resolve to overlapping physical directories. Do this with Windows-resolved path containment checks in the service transaction, not only string prefixes. Re-registration of the *same* existing root with the *same* class is idempotent and returns the existing root ID.
- A `--work-map` CSV is scoped to **one selected root**. No cross-root ambiguous relative names; import each root separately. The CSV supplies only `relative_path,work_id`, never a classification column. The selected work must be a leaf whose class is compatible with the registered root. For example, importing a `fanfiction` root into an `episode` work is rejected, even if the text looks like a script.
- On read-only inventory, show each direct child file's `root_id`, `relative_path`, `source_class`, file format, and optional proposed work assignment. A mismatch between operator folder conventions and registration should be corrected by the operator before ingestion; never automatically reclassify documents.
- For a single universe in the pilot, canonical scripts and fanfiction works may share the same **series parent** while remaining separate leaf works and classes. Multiple universes can use separate series parents, and search continues to require explicit `work_id` scoping or explicit `--all-works`.

**Example flat directory layout (illustrative, not required):**

```text
D:\ScriptSources\
  Scripts\
    Episode_01.txt
    Episode_02.html
  Fanfiction\
    Story_01.htm
    Story_02.txt
```

**Integrity requirement:** Add an SQLite `source_roots.source_class` constraint, plus trigger/service checks preventing a slot or document whose class differs from its root. Tests must attempt to bypass classification through direct SQL writes and verify rejection; root overlap remains a filesystem/service-level constraint.

## 4. Supported inputs and normalization

**Required in v1:** `.txt`, `.html`, `.htm` **only** (case-insensitive extensions). Each registered flat root contains supported files directly; nested directories, symlinks, and junctions are unsupported. If encountered, report the offending path and fail inventory/ingestion before writing records. Reject every other extension with a reported reason; do not attempt PDF, DOCX, EPUB, archive reading, OCR, or external downloads. For HTML, extract visible text using an HTML parser while keeping paragraph, line-break, heading and dialogue-relevant structure; ignore scripts/styles/navigation where possible and do not execute HTML or fetch linked resources. For TXT, preserve original structure. Report detected/selected encoding and decode warnings. Source files are read-only.

Implement format adapters with a common contract:

```python
@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    source_format: str
    locator_map: list[dict]  # optional page/paragraph/line anchors
    extraction_warnings: list[str]
```

Normalization rules must be explicit and versioned (`normalizer_version`): preserve meaningful line breaks, normalize line endings to `\n`, use consistent Unicode handling, and avoid transformations that destroy speaker labels/indentation. Do not lower-case or strip punctuation from stored source text. Extraction is not character analysis.

For each source file, store absolute source path, relative path from import root, registered `root_id` and its inherited `source_class`, basename, extension, size, SHA-256 of raw bytes, normalized text SHA-256, timestamps and errors. SHA-256 is for duplicate detection, **not proof that two differently formatted files represent different creative works**.

Never ingest files from internal `data/` or generated output directories. Reject symlinks, junctions, nested directories, and registered source roots that overlap each other before any writes; do not follow or traverse links. Treat file extension checks case-insensitively. Directory names are illustrative, not magical names: only root registration determines source class. Include sane maximum file-size protection with a documented override.

## 5. Canonical database design: smallest useful foundation

Use SQLite with `PRAGMA foreign_keys=ON` on every connection. Store normalized text as a durable source for citations and rebuilds. Use one initial schema migration, tested from an empty DB.

### Required relational entities

| Entity | Minimum purpose |
|---|---|
| `works` | Logical creative work with UUIDv4 `work_id`, constrained `kind`, `parent_work_id`, and leaf-only `source_class`; one episode may have several representations |
| `source_roots` | Canonical Windows import-root identity and normalized root key |
| `source_slots` | Unique `(root_id, relative_path_key)` path identity, assigned work/source class, plus FTS and hybrid published document pointers |
| `document_visibility` | **Optional name for the same source-slot pointer contract**; do not create two competing visibility authorities |
| `documents` | One immutable imported file snapshot (`document_id` UUIDv4), `source_slot_id`, `work_id`, fixed `source_class`, SHA-256, status, optional `supersedes_document_id`; do not overwrite content |
| `document_texts` | One normalized text and normalization version for each document; source offsets refer here |
| `scenes` | Ordered structural scene records with `[start_cp,end_cp)` offsets in document normalized text |
| `characters` | Resolved character identity within a universe; initial values can be provisional |
| `character_aliases` | Original speaker label mapped to canonical identity, with review status |
| `dialogue_turns` | Speaker label/optional resolved character, scene ID, dialogue offsets, order and attribution confidence |
| `chunks` | Passages for retrieval (`dialogue_exchange`, `scene`, `prose_passage`) with source offsets and stable IDs |
| `index_jobs` | One job per `(chunk_id, index_version)` with retries, expected vector ID, error and verified status |
| `index_versions` | Embedding model ID, embedding dimension, chunk schema, Chroma collection name, verified/active flags |

**Offset convention:** half-open Unicode code-point offsets in the exact normalized text string stored in SQLite: `[start_cp,end_cp)`. For a stored span, `document_text[start_cp:end_cp]` must reconstruct the intended passage. Keep the normalizer version recorded.

### Integrity invariants

- Valid source classification: `canonical` or `fanfiction` (do not mix `official` and `canonical` strings).
- Each scene belongs to exactly one document; a dialogue turn's scene and document match.
- All scene, dialogue, and chunk offsets are nonnegative, ordered, and within the normalized text length.
- A dialogue turn's span is within its scene where the scene is known.
- `chunks` reference one document and optionally one scene; no cross-document scene association.
- Re-importing the same file (raw SHA-256 + stable path/import identity) does not create duplicate records.
- Reject mismatched scene/doc and out-of-bounds offsets with SQL triggers and/or a single audited transactional validation layer; **include raw-SQL negative tests** for the invariants that the database promises to enforce.
- If a file is already present with the same path but changed raw hash, **record a conflict and do not silently replace it**. The exceptional resolution procedure in §9.3 is operator-driven and retains the old source as a historical record.
- Use `work_id` to support multiple source representations of the same episode; create/assign IDs using the CLI contract above. Never merge works based on filenames or hash guesses. Test cross-work FK violations and contradictory work/source-class assignment.

Use a limited processing `status` (`importing`, `fts_ready`, `hybrid_ready`, `failed`, `archived`) and per-chunk vector-job state. **Status is not the authority for search visibility:** published pointers in `source_slots` are, as specified in §9.1 and Appendix B. `fts_ready` and `hybrid_ready` describe eligibility, not automatic publication. This is **not** a general-purpose multi-version lifecycle system.

## 6. Script parsing contract

This parser should be deterministic and conservative. Preserve unrecognized text, flag uncertain parsing, and record provenance.

Detect when formatting supports it:

- Scene headings (`INT.`, `EXT.`, location/time headings; configurable variants).
- Act/teaser headings when present.
- Dialogue speaker labels and consecutive turns.
- Parentheticals, transitions and stage/action directions.
- Scene ordering and location/time metadata when reliable.

A speaker label is **not** a character identity until resolved. Preserve source label (e.g. `BUFFY (O.S.)`) and map aliases separately (e.g. `BUFFY`). Do not treat ordinary capitalized prose lines as dialogue by default. Support review flags for uncertain boundaries or labels.

**Crucial:** Use source spans; do not reconstruct quotations from model summaries. The exact text must remain retrievable.

## 7. Fan-fiction parsing contract

Fan-fiction prose is not formatted like screenplays. Segment primarily by chapter markers, scene breaks, paragraph groups and coherent text windows. Attributed speech is optional.

- Store narrative prose passages without requiring a speaker.
- Identify quoted utterances only when textual conventions are reliable.
- If a speaker is not clearly attributed, set `speaker_character_id=NULL`, record the original nearby text, and mark low confidence.
- Do not invoke Qwen3:8b to invent missing speakers in this first phase.
- Preserve chapter headers and story-level metadata when available, without guessing canon era or relationships.

Later analysis can revisit ambiguous passages using a model, but the raw source reference must remain intact.

## 8. Chunking strategy and vector collections

Do **not** embed arbitrary 1,000-character windows across dialogue boundaries. Chunk by meaning and document structure.

### Chunk types

1. `dialogue_exchange`: contiguous dialogue turns, usually a short exchange; include neighboring turns within a bounded window, while preserving the individual speakers and turn IDs.
2. `scene`: a complete short scene or bounded segments of a long scene; use ordered overlap without crossing document boundaries.
3. `prose_passage`: coherent fan-fiction passage split at paragraph/scene boundaries.

Use a configurable target of roughly 300–700 tokens per semantic passage and up to about 100 tokens of overlap for long segments, but **do not compromise speaker attribution or source offsets just to hit the token target**. Store precise membership and boundaries. Scene metadata may be attached separately from the text embedded.

Use two initial Chroma collections, or equivalent partitions:

- `scriptlab_dialogue_v1`: `dialogue_exchange` chunks.
- `scriptlab_passages_v1`: `scene` and `prose_passage` chunks.

A third future collection for character/narrative claims is **not** populated in this phase.

Each vector record should contain deterministic `chunk_id` and minimal metadata: `document_id`, `source_class`, `chunk_type`, `work_id`, `scene_id` if any, start/end offsets, and embedding model/index version. SQLite owns the full text and canonical metadata.

### Embedding interface

Define an adapter, not direct Ollama calls scattered through indexing code:

```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def model_id(self) -> str: ...
```

Default to Ollama model `qwen3-embedding:0.6b`. Keep the embedding model identifier, dimension, and index version in persistent configuration/metadata. Reject mixed-dimension or mismatched-model embeddings within one collection; rebuild a new collection when the model changes. Process small batches on 8 GB VRAM. Do not store separate raw vector arrays in SQLite unless needed for diagnostic purposes.

## 9. Index visibility, restart safety, and exceptional conflict resolution

### 9.1 Visibility contract — published pointers, distinct modes

**Chosen policy:** Every registered path has one `source_slots` row. Its `fts_document_id` and `hybrid_document_id` are independently published, nullable document pointers. Both pointers must reference a document attached to **that slot**, and each pointer is set only when that document meets the corresponding readiness invariant. Search visibility is always determined by these pointers, not merely by `documents.status` or by the presence of vectors. Queries join the `source_slots` pointer and current active `index_versions` through SQLite. `FTS` is a transactional SQLite index; Chroma is subordinate and all hits are hydrated/validated through SQLite.

| Mode | Candidate source(s) | Eligible published pointer | Embedding service required? | Fusion |
|---|---|---|---|---|
| `fts` | **FTS5 only** | `fts_document_id` | No | No; use FTS rank |
| `semantic` | **Chroma only** | `hybrid_document_id` | Yes, to embed query | No; use vector similarity rank |
| `hybrid` | **FTS5 + Chroma** | `hybrid_document_id` for **both** sources | Yes | Yes, RRF |

`semantic` never includes FTS-only results and never uses keyword scores. `hybrid` never supplements its vector results with FTS-only documents. If Ollama/Chroma is unavailable for `semantic` or `hybrid`, **fail with `VECTOR_UNAVAILABLE` by default**. The explicit `fallback_to_fts=True` or `--fallback-to-fts` option changes that behavior to an FTS-only query using the `fts_document_id` pointer and a **machine-readable** `actual_mode='fts'`, `fallback_reason`, plus a CLI warning. Fallback is used only for vector-service unavailability or inability to embed the query—not as a way to fill low-match or partly indexed search results. In `fts` mode the fallback option is accepted but has no effect and no warning. Never label fallback output as semantic/hybrid.

For an initial import, publish the FTS pointer **in the same SQLite transaction** that commits and validates all normalized text, scenes, chunks and FTS rows. At this stage `hybrid_document_id=NULL`. After **all** expected vector IDs for the target document and active index version are verified in Chroma, atomically publish the hybrid pointer in SQLite. If any vector fails, leave the hybrid pointer unchanged; FTS remains available. An index rebuild targeting a new `index_version` writes to a new Chroma collection and switches the active index version only after full verification of its published hybrid set (otherwise retain the old active version). This is a scoped release mechanism, not a general document-version lifecycle.

For an explicit same-path replacement, `fts_document_id` can switch to the new document before `hybrid_document_id`, which stays on the previous fully vectorized document. Thus `fts` and `hybrid` may temporarily observe **different snapshots** of that path; return `document_id` and `source_hash` in every result, and show a `replacement_pending_vectors` warning to explain the mismatch. On hybrid readiness, switch `hybrid_document_id` atomically to the new document. The two modes must never return both the old and new document **within one mode for the same slot**. `semantic` follows the hybrid pointer. Historical documents are accessible only through diagnostic/history commands.

**Publication invariants:** a pointer can only reference the same slot; FTS publication requires complete/validated FTS records; hybrid publication requires verified vectors for every expected chunk in the active index version; pointer changes are SQLite transactions; search rechecks the pointer after candidate retrieval and before returning results. A failed replacement cannot remove the previous published pointer. See Appendix B for schema and SQL transaction predicates.

### 9.2 Restart safety

- Commit normalized text, metadata, chunks, and FTS rows in SQLite transactions with stable identifiers.
- Keep `index_jobs` for each expected chunk and model/index version; use deterministic vector IDs and idempotent upserts.
- On restart, reconcile expected chunk IDs against Chroma, and resume incomplete jobs. A vector write without its acknowledged SQL job must be safe to repeat.
- `scriptlab validate` checks FTS records, expected-vector counts, orphan vector IDs, and status inconsistency; failed checks prevent `hybrid_ready` promotion.
- No general-purpose distributed transactions or forensic deletion workflow is needed for this milestone.

### 9.3 Operator workflow for changed files at the same path

Normal imports are fixed-corpus. A changed SHA-256 at an already imported path yields a **conflict**, not replacement. Implement the exceptional commands:

```powershell
scriptlab conflicts list
scriptlab conflicts show <conflict-id>
scriptlab conflicts resolve <conflict-id> --strategy keep-old
scriptlab conflicts resolve <conflict-id> --strategy archive-and-import --confirm
```

`conflicts show` displays old/new hashes, path, work assignment, source class, file sizes, and extraction warnings; it never modifies source files. `keep-old` closes the conflict without changing the indexed document (a subsequent changed import may report it again). `archive-and-import` must first re-read and verify that the new file still matches the conflict's recorded hash and that the selected work and source class remain valid. Create a **new document row with a new `document_id`**, retain the previous record/text as historical and read-only, and link it using `supersedes_document_id`. Do not mutate the old text, offsets or citations.

The old document remains searchable while the new document is `importing` or `fts_ready` (for hybrid queries). On **FTS publication**, atomically switch FTS visibility from old to new for FTS queries. On **hybrid publication**, atomically switch hybrid visibility from old to new for hybrid/semantic queries. Model this explicitly using a `document_visibility` table or equivalent atomic SQLite pointers per `source_path_identity` and search mode; do **not** rely on a single status flag that would simultaneously hide the old version from hybrid before the replacement vectors are ready. If replacement FTS or vector indexing fails, preserve the last working document for that search mode and show the failed replacement in `status`. Archived history is available only through an explicit diagnostic/history command, not regular retrieval.

This is a narrow, deliberate replacement escape hatch, not a general automatic update system. If its complexity threatens the first milestone, implement `conflicts list/show/keep-old` now and **mark archive-and-import as a gated follow-up**; do not claim replacement support is complete until its atomic visibility tests pass.

## 10. Retrieval contract

Implement:

```python
def search(
    query: str,
    *,
    work_id: str | None = None,
    include_descendants: bool = False,
    source_class: str | None = None,
    kind: str | None = None,  # chunk type: dialogue_exchange | scene | prose_passage
    limit: int = 10,
    mode: Literal["fts", "semantic", "hybrid"] = "hybrid",
    all_works: bool = False,
    fallback_to_fts: bool = False,
) -> SearchResponse: ...
```

Exactly one of `work_id` or `all_works=True` is required, with no implicit global search. `include_descendants=True` requires a series `work_id` and cannot be combined with `all_works=True`. CLI uses `--mode {fts,semantic,hybrid}` and `--fallback-to-fts` with the same semantics. Return `requested_mode`, `actual_mode`, `fallback_reason` (nullable), `partial_scope_warning`, and entries with citation IDs.

1. Resolve/validate the work scope and source-class filter **before retrieval**. FTS queries select FTS candidates only in `fts` mode; vector queries select Chroma candidates only in `semantic` mode; `hybrid` invokes both over the same `hybrid_document_id`/active-index snapshot.
2. Apply the same work, source-class, chunk-type, and published-pointer filters to all candidates. FTS uses published chunk/document joins; Chroma returns IDs, and SQLite filters and hydrates them. Overfetch vector candidates when filtering so scope restrictions do not create false matches; never trust Chroma metadata alone.
3. Apply Reciprocal Rank Fusion with configurable `k=60` **only** for hybrid (`sum(1/(60+rank))` across present ranking sources). Deduplicate by `chunk_id`. `fts` and `semantic` preserve their respective source order, with a deterministic tie-break on `chunk_id`.
4. Validate citation IDs and exact offsets in SQLite. Return work title, `work_id`, source classification, document ID, SHA-256, source path, offsets, chunk type, scene/speakers if known, excerpt and scoring/ranking provenance.
5. Default errors for unavailable vector query/index services are explicit (`VECTOR_UNAVAILABLE`), never silent fallback. Explicit `fallback_to_fts` performs **one FTS-only query**, clearly labeled. Warn when selected works contain FTS-published but not hybrid-published slots, even if some hybrid results are available.

No LLM synthesis is required: this is retrieval, not chat generation.

## 11. Tests and evaluation

All unit tests must use **synthetic fixtures committed with the repository**, not copyrighted episode scripts. A separate ignored pilot corpus can be evaluated manually.

### Automated tests

- `test_schema`: foreign keys active; invalid scene/doc or span relationships rejected; text offsets valid; classifications constrained; invalid source-slot/pointer ownership rejected by raw SQL.
- `test_extraction`: `.txt`, `.html`, `.htm` only; visible HTML text extraction and preserved block structure; non-UTF-8 decoding/warnings; no PDF/DOCX codepaths; reproducible normalization.
- `test_script_parser`: known scene headings, speaker labels, directions and ambiguous all-caps lines.
- `test_fanfiction_parser`: chapter headings, prose chunks, quoted dialogue with unknown speaker.
- `test_ingestion`: repeat import is idempotent; same-path modified file becomes a recorded conflict; interruptions resume without duplicate records.
- `test_indexing`: fixed fake embedding adapter for reproducibility; interrupted vector upsert replays safely; model/dimension mismatch prevented.
- `test_works`: valid/invalid work ID creation; same work multiple representations; multi-work CSV mappings; reject unknown, malformed, cross-work or conflicting source assignments; validate parent-child work expansion.
- `test_visibility`: exact source-slot pointer transitions; FTS publication while embeddings pending; no semantic/hybrid visibility before all vectors verify; failed index leaves FTS pointer intact; old/new replacement pointers switch separately; crashes before/after pointer commit; no mixing active index versions.
- `test_conflicts`: changed file creates conflict, keep-old behavior, changed-again hash rejection, and (when implemented) archive-and-import with mode-specific atomic visibility and rollback on failure.
- `test_search`: FTS-only, semantic vector-only, hybrid fusion, `fallback_to_fts` API and CLI, explicit `requested_mode`/`actual_mode`, failure without fallback, required work scope, descendants, all-works opt-in, citations and source filters.
- `test_path_identity`: Windows case, slash/backslash, NFC, `.`/`..`, symlink/junction rejection, nested-directory rejection, root identity, duplicate mapping keys, identical bytes at distinct paths, and same-path hash conflict.
- `test_source_roots`: separate script/canonical and fanfiction flat roots; reject nested/overlapping roots and duplicate aliases with conflicting classes; verify inherited classification for direct child files; refuse `--source` for import; do not import unsupported formats, nested paths, or files outside root.
- `test_work_hierarchy`: source classes only on leaves, invalid parent/child kinds, series direct import forbidden, `fanfic`/`episode` fixed classes, `other` first-import class lock and rollback.
- `test_uniqueness`: repeated same-path imports and same-index retries do not duplicate documents/chunks/index jobs; distinct paths can share hash without deduplication.

Run:

```powershell
python -m pytest -q
scriptlab validate
```

### Manual pilot review

Create `pilot_review.csv` *outside source control* with fields: `document_id`, `scene_id`, `turn_id`, `expected_speaker`, `observed_speaker`, `parse_issue`, `reviewer`, `notes`. Review examples from different formats, seasons, and authors.

Create `retrieval_gold.csv` with at least ten fixed natural-language questions, expected relevant source IDs, and judgments `2=direct`, `1=partial`, `0=irrelevant`. A *top-five hit* requires a score-2 passage among the first five results. Report Hit@5 and example misses; treat the sample as directional, not a statistically decisive benchmark.

**Pilot targets (provisional):**

- Twenty pilot documents import with clear per-file success/failure reporting.
- On clearly formatted script dialogue, aim for at least **95% correct attribution** on a manually checked sample; keep ambiguous prose explicitly unassigned.
- At least **8 of 10** curated queries have a directly relevant result in the top five.
- Every returned search hit has a working SQLite citation and correctly reconstructible source text.
- Re-running import/index changes no counts for unchanged sources.
- Kill and restart indexing once; confirm no duplicate chunks/vectors, FTS-only availability during interruption, and no partially indexed document in hybrid results.
- Search two deliberately separate fictional works with similar names; prove a query scoped to Work A **never** returns Work B. Test series-with-descendants separately.

## 12. Implementation sequence for Claude Code

Do not attempt the entire specification in one undifferentiated coding pass. Complete and test each work package before moving on.

| Package | Work | Gate |
|---|---|---|
| **P1** | Repo, root configuration, CLI argument skeletons (`source-root` and root-scoped import), command stubs, TXT/HTML fixtures, README | `scriptlab --help` and parser-level CLI contract tests pass; **no SQLite, visibility behavior or DB-backed work creation** |
| **P2** | SQLite schema, source-root registration/class invariants, work commands, path identities, source slots, pointer constraints, conflicts | Integrity/negative SQL tests, work-ID tests, and *schema-level* visibility-transition tests pass |
| **P3** | Root-scoped flat TXT/HTML inventory and format extraction | All supported fixtures extract; nested directories/links fail clearly; originals untouched |
| **P4** | Script/fan-fiction parsers and source offsets | Parsing fixture assertions and human-readable report pass |
| **P5** | Transactional ingest, mappings, conflicts, FTS5, FTS publication | Idempotency and FTS pointer/visibility integration tests pass offline |
| **P6** | Ollama embedding adapter and Chroma index | Fake and live indexing smoke tests pass |
| **P7** | Work-scoped retrieval modes, fallbacks, hybrid publication, citations, rebuild, optional conflict replacement | Search-mode/transition/recovery integration tests pass |
| **P8** | Twenty-document pilot audit and handoff | Report failures, benchmarks and next-phase limitations |

At each gate, Claude Code must report: files modified, commands run, passing/failing tests, known issues, and the proposed next package. Don't silently change the source taxonomy or add hosted networking.

## 13. Non-goals and guardrails

**Do not implement in this phase:**

- Qwen3:8b extraction of motivations, relationships or narrative conventions.
- Character dossiers or continuity event inference.
- Prompting a hosted model, MCP server, or cloud upload.
- Fine-tuning, autonomous episode generation or a web UI.
- Granular source-rights tables, deletion/purge workflows, or automatic source replacement. A narrow manual same-path conflict resolution is permitted as defined in §9.3.
- Automatically treating fan-fiction assertions as canonical facts.

**Do implement:** Source provenance, bounded extraction errors, correct joins and offsets, reproducible indexing, clear parser uncertainty, and honest retrieval evaluation. Treat any source text as data, never as instructions to the indexing application.

## 14. Handoff into the next milestone

When this phase passes, the **Character Intelligence Specification** can rely on stable APIs such as:

```python
get_dialogue_turns(character_id: str, work_id: str, source_class: str | None = None)
get_scene(scene_id: str)
get_source_span(document_id: str, start_cp: int, end_cp: int)
search(query: str, work_id: str, source_class: str | None = None, kind: str | None = None)
```

The next build adds evidence-backed character traits, relationship-conditioned behaviors, temporal change, and eventually the narrative model. No implementation in this phase should hard-code a particular character or fictional universe.

---

# Copy/paste kickoff prompt for Claude Code

> We are beginning ScriptLab's first implementation milestone. Read `SCRIPTLAB_CORPUS_INGESTION_AND_INDEXING_SPEC_v1.3.md` in full. Also read `ScriptLab_Revised_Project_Vision_v0.2.md` if available. The aim is a source-linked local corpus and hybrid retrieval foundation, not character generation yet.
>
> Implement **P1 only**: create the Python project structure, dependencies, configuration, CLI skeleton, synthetic fixtures, and initial tests for Windows. Do not yet implement indexing or call Ollama. Keep source categories exactly `canonical` and `fanfiction`; the project owner explicitly allows unrestricted **local application use** for this pilot, with **no external transmission**. Declare `source-root add/list`, `inventory --root/--all-roots`, `work create`, `ingest --root/--file/--work-id/--work-map/--create-works-per-file`, and `search --work-id/--mode/--fallback-to-fts` as **argument-parser stubs only**. Reject `--source` as an import option (it remains valid for search filtering). P1 tests check that these CLI options parse and show help, not that database-backed commands or index visibility work. Implement root registration, work persistence, source slots and schema-level pointer tests in P2; functional FTS visibility in P5; vector/hybrid visibility and search-mode behavior in P6–P7. Do not assume this is a license grant. Do not build deletion, elaborate permissions, hosted-model integration, or MCP.
>
> Before coding, list your intended files and commands. After coding, run the tests, report the results and changed files, and stop. Ask for permission to proceed to P2 rather than generating the entire application in one pass.

## Appendix A — Review-response traceability (v1.1 + v1.2)

| Review finding | Resolution | Verification |
|---|---|---|
| 1. Import/work contract | `work create/list/show`; required `--work-id`, `--work-map`, or `--create-works-per-file` at ingestion; validation and multi-representation mapping | `test_works`, bulk mapping dry run |
| 2. Work-scoped search | Mandatory `work_id` or explicit `all_works`; descendant scope for series; no accidental cross-work hits | `test_search`, two-work isolation fixtures |
| 3. Permissions/scope conflict | Owner-selected unrestricted **local** use explicitly overrides earlier local permission gating for this pilot; external egress absent; legal/provider rights not inferred | No hosted client/network calls; config and policy tests |
| 4. Search publication | FTS-ready after committed parse/FTS; hybrid-ready only after all required vectors verified; mode-aware published visibility | `test_visibility`, outage and partial-upsert tests |
| 5. Fixed-corpus conflict | Conflict inspection, keep-old, and explicit archive-and-import with old records preserved and mode-specific atomic publication; gated if necessary | `test_conflicts`, rollback/crash tests |

**Document precedence:** This v1.3 specification governs the initial corpus ingestion/indexing milestone, superseding v1.1 and v1.2 where they differ. The Revised Project Vision v0.2 governs product goals. Earlier hardening guidance remains useful for structural integrity and reproducibility; its extensive permissions, purge, and external-approval workflows are consciously deferred under the owner's personal local-only pilot scope.

## Appendix B — Normative IDs, uniqueness, source-slot schema, and atomic publication (v1.2)

**Stable identity and uniqueness rules:**

| Entity | Identifier | Mandatory unique key / deduplication semantics |
|---|---|---|
| `works` | Random UUIDv4 `work_id`, persisted | Primary key only; different works may share titles; no filename/hash-based merging |
| `source_roots` | UUIDv4 `root_id`, persisted; fixed `source_class` | `UNIQUE(root_key)`, constrained class, no nested/overlapping resolved roots |
| `source_slots` | UUIDv4 `slot_id`, persisted | `UNIQUE(root_id, relative_path_key)` and `UNIQUE(source_path_identity)` |
| `documents` | UUIDv4 `document_id` for each imported snapshot | `UNIQUE(source_slot_id, raw_sha256)` for idempotent exact-content reimport at that path; a deliberate replacement with different hash creates a second document at same slot |
| `scenes` | Deterministic `SHA256(document_id, parser_version, ordinal)` | `UNIQUE(document_id, parser_version, ordinal)`; parser changes create new extraction revision, not silent overwrite |
| `chunks` | Deterministic `SHA256(document_id, chunk_schema_version, chunk_type, start_cp, end_cp, ordinal)` | Unique tuple `(document_id,chunk_schema_version,chunk_type,start_cp,end_cp,ordinal)`; ensure no duplicate IDs or types |
| `index_versions` | UUIDv4 `index_version_id` | `UNIQUE(collection_name)` and model+dimension+chunk-schema metadata validated per collection |
| `index_jobs` | Deterministic `SHA256(chunk_id, index_version_id)` | `UNIQUE(chunk_id,index_version_id)`; vector ID is same deterministic job ID, safe to upsert |
| `work-map` CSV keys | `relative_path_key` within specified import root | `UNIQUE(relative_path_key)` per map; reject case/Unicode-equivalent duplicates, including those with conflicting work IDs |
| `import_conflicts` | UUIDv4 `conflict_id` | At most one open conflict per `(slot_id, proposed_raw_sha256)`; repeated detection updates observation count, not duplicate conflicts |

UUIDs and deterministic hashes must use lowercase canonical hex text; UUID input is parsed and normalized before comparison. Implement `sha256` over an unambiguous length-prefixed or canonical JSON serialization of components, **not** raw string concatenation with ambiguous delimiters. An import can be repeated at the same slot only when work/source class match the existing slot; differing assignments are operator conflicts. An identical raw hash in a different slot is a different document. Reprocessing an unchanged document with a changed parser/chunker version is a separately identified *derived extraction revision* and must not overwrite source spans; an implementation may rebuild all derived records in a coordinated offline maintenance operation rather than add an incremental extraction-version lifecycle in v1.

**Reference source-slot/pointer contract:** (illustrative SQL; implementation must complete keys, migration order and triggers)

```sql
CREATE TABLE source_slots (
  slot_id TEXT PRIMARY KEY,
  root_id TEXT NOT NULL REFERENCES source_roots(root_id),
  relative_path_key TEXT NOT NULL,
  source_path_identity TEXT NOT NULL UNIQUE,
  work_id TEXT NOT NULL REFERENCES works(work_id),
  source_class TEXT NOT NULL CHECK(source_class IN ('canonical','fanfiction')),
  fts_document_id TEXT,
  hybrid_document_id TEXT,
  UNIQUE(root_id, relative_path_key),
  UNIQUE(slot_id, work_id, source_class)
);

CREATE TABLE documents (
  document_id TEXT PRIMARY KEY,
  source_slot_id TEXT NOT NULL,
  work_id TEXT NOT NULL,
  source_class TEXT NOT NULL,
  raw_sha256 TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN
    ('importing','fts_ready','hybrid_ready','failed','archived')),
  supersedes_document_id TEXT REFERENCES documents(document_id),
  UNIQUE(source_slot_id, raw_sha256),
  UNIQUE(document_id, source_slot_id),
  FOREIGN KEY(source_slot_id,work_id,source_class)
    REFERENCES source_slots(slot_id,work_id,source_class)
);
```

Add **composite foreign keys** from `(fts_document_id, slot_id)` and `(hybrid_document_id, slot_id)` to `(document_id, source_slot_id)` using an appropriate schema layout, along with triggers that reject publishing unless all readiness conditions hold. Because the reference table-order and composite FK syntax need careful construction, the final P2 migration is authoritative; tests must create direct SQL attempts at cross-slot pointer assignment and verify rejection. If a `source_slots` row exists without documents, both pointers are NULL. Publishing an FTS pointer requires all required chunks and FTS postings verified, and the document's `status IN ('fts_ready','hybrid_ready')`. Publishing a hybrid pointer requires all expected vector jobs verified for the **active** index version and the document's `status='hybrid_ready'`. As readiness verification involves Chroma, the publisher must verify Chroma, then issue a conditional SQLite transaction (compare expected old pointer and active index version); if condition fails, abort and recheck. No query should rely on a non-transactional status check alone.

**Atomic FTS publication:** in one SQLite transaction: assert source slot's current `fts_document_id` still equals the expected previous ID, commit normalized text/parsed records/FTS postings, validate expected counts and ranges, set `status='fts_ready'`, set `fts_document_id` to new ID, commit. If any step fails, rollback and preserve old pointer. Do not publish individual chunks before the entire document passes validation.

**Atomic hybrid publication:** after verifying vector IDs/count, use a single SQLite transaction to assert the source-slot's `hybrid_document_id` matches the expected old ID, the active `index_version_id` matches the verified version, every index job is verified, and the target belongs to the slot; set `status='hybrid_ready'` and switch `hybrid_document_id`. A crash before commit preserves old hybrid visibility; a crash after commit exposes new complete content. Chroma vectors are never authoritative; any orphan/partial vector remains unsearchable until SQLite publication and can be reconciled later. On reindex of a *new* index version, use one atomic global active-index switch after validating the entire published set. Keep old collection until the switch is complete.

**Replacement failure cases:** After successful FTS switch but failed embedding, leave `fts_document_id=new` and `hybrid_document_id=old`; warn about mixed snapshots. If FTS publication fails, both pointers stay on old. After a successful hybrid switch, both point to new; old document remains retained as history. A repeated resolver call with the same conflict ID is idempotent and cannot switch pointers backward. `status` is informational and may reflect `failed` for a staged replacement, but **never mark a currently published old snapshot failed merely because the new one failed**.

**Required negative tests** include: null/missing pointer; pointer to document in another slot; pointer to not-fully-indexed document; stale expected pointer; wrong active index version; vector failure midway; simulated crash before and after each publication transaction; FTS and hybrid queries returning distinct versions during replacement; two workers trying to publish different snapshots; and an all-works search whose results are properly labeled as cross-work.

## Appendix C — v1.3 source-layout decision and test traceability

| Confirmed source decision | Required implementation | Verification |
|---|---|---|
| Ordinary filesystem directories | Register disjoint Windows roots; retain root-relative paths | `test_source_roots`, `test_path_identity` |
| Only text or HTML | Accept `.txt`, `.html`, `.htm`; no other format adapter | `test_extraction`, unsupported-format cases |
| Canon and fanfiction in separate directories | Inherit immutable class from root; prohibit per-file class override | SQL class mismatch negative tests, CLI parser test |
| Flat source roots | Scan direct child files only; nested folders and links fail clearly | Flat-root inventory and rejection tests |
| Existing work mappings still apply | Per-root work map, leaf compatibility checks | `test_works`, `test_work_hierarchy` |

**v1.3 supersedes v1.2 where file formats, import arguments, fixtures, or source-class assignment conflict; all other v1.2 contracts remain in force.**

## Appendix D — Prior v1.2 review response matrix

| Concern | Normative section(s) | Test coverage |
|---|---|---|
| Search mode meaning and fallback | §9.1, §10 | `test_search_modes`, `test_search`, service-down/fallback tests |
| Windows paths and same-hash different-path | §3.4, Appendix B | `test_path_identity`, duplicate handling |
| Work kind/class invariant | §3.3, Appendix B | `test_work_hierarchy`, negative SQL tests |
| Atomic visibility pointers | §9.1, Appendix B | `test_visibility`, crash fault injection |
| IDs and uniqueness | §3.4, Appendix B | `test_uniqueness`, idempotent job tests |
| P1 vs P2 sequencing | §12, kickoff prompt | P1 CLI parser-only smoke tests; P2 schema state tests |

**Change control:** This v1.3 supersedes v1.2 for the ingestion/indexing phase. It does not change the product goal, local-only policy, source taxonomy, or deferred premium/character-analysis work. Claude Code must not treat these appendices as optional guidance.
