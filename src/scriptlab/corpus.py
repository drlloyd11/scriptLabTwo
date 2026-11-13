"""Safe flat-folder inventory and text extraction for Step 2."""

import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag

from scriptlab.config import ProjectConfig


SUPPORTED_EXTENSIONS = frozenset((".txt", ".html", ".htm"))
_REPARSE_POINT = 0x0400
_BLOCK_TAGS = frozenset(
    (
        "address",
        "article",
        "blockquote",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    )
)
_REMOVED_TAGS = ("script", "style", "nav", "noscript", "template")


class CorpusError(ValueError):
    """Raised when a source or output path cannot be safely used."""


class FlatFolderError(CorpusError):
    """Raised when a configured source root is missing or not flat."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__(
            "Source-folder preflight failed; no source files were read:\n- "
            + "\n- ".join(self.issues)
        )


class ExtractionError(CorpusError):
    """Raised when source bytes cannot be decoded or contain suspicious text."""

    def __init__(self, message: str, encoding: Optional[str] = None) -> None:
        self.encoding = encoding
        super().__init__(message)


@dataclass(frozen=True)
class SourceRoot:
    path: Path
    source_class: str


@dataclass(frozen=True)
class SourceFile:
    path: Path
    source_class: str
    file_format: str
    size_bytes: int


@dataclass(frozen=True)
class UnsupportedFile:
    path: Path
    source_class: str
    file_format: str
    size_bytes: int


@dataclass(frozen=True)
class SourceScan:
    roots: Tuple[SourceRoot, ...]
    files: Tuple[SourceFile, ...]
    unsupported: Tuple[UnsupportedFile, ...]


@dataclass(frozen=True)
class InventoryFile:
    path: Path
    source_class: str
    file_format: str
    size_bytes: int
    sha256: str
    status: str
    encoding: Optional[str] = None
    extracted_characters: Optional[int] = None
    error: Optional[str] = None
    recovery_warnings: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "filename": self.path.name,
            "path": str(self.path),
            "source_class": self.source_class,
            "format": self.file_format,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "status": self.status,
            "encoding": self.encoding,
            "extracted_characters": self.extracted_characters,
            "error": self.error,
            "recovery_warnings": list(self.recovery_warnings),
        }


def _configured_roots(config: ProjectConfig) -> Tuple[SourceRoot, ...]:
    roots = tuple(
        SourceRoot(path=path, source_class="canonical")
        for path in config.canonical_directories
    ) + (SourceRoot(path=config.fanfiction_directory, source_class="fanfiction"),)
    return roots


def _is_reparse_point(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _assert_path_has_no_reparse_components(
    path: Path, allow_missing_leaf: bool = False
) -> Optional[os.stat_result]:
    absolute_path = Path(os.path.abspath(os.fspath(path)))
    parts = absolute_path.parts
    if not parts:
        raise CorpusError("Invalid empty path.")

    current = Path(parts[0])
    final_info: Optional[os.stat_result] = None
    for index, component in enumerate(parts[1:], start=1):
        current = current / component
        is_leaf = index == len(parts) - 1
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf:
                return None
            raise CorpusError("Path does not exist: {!s}".format(current))
        except OSError as exc:
            raise CorpusError(
                "Could not inspect path {!s}: {}".format(current, exc)
            ) from exc

        if _is_reparse_point(info):
            raise CorpusError("Symlinks and junctions are not allowed: {!s}".format(current))
        if not is_leaf and not stat.S_ISDIR(info.st_mode):
            raise CorpusError("Path component is not a directory: {!s}".format(current))
        final_info = info
    if len(parts) == 1:
        try:
            final_info = os.lstat(current)
        except OSError as exc:
            if allow_missing_leaf and isinstance(exc, FileNotFoundError):
                return None
            raise CorpusError(
                "Could not inspect path {!s}: {}".format(current, exc)
            ) from exc
        if _is_reparse_point(final_info):
            raise CorpusError(
                "Symlinks and junctions are not allowed: {!s}".format(current)
            )
    return final_info


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _paths_overlap(first: Path, second: Path) -> bool:
    first_path = _normalized_path(first)
    second_path = _normalized_path(second)
    try:
        common = os.path.normcase(os.path.commonpath((first_path, second_path)))
    except ValueError:
        return False
    return common == first_path or common == second_path


def scan_sources(config: ProjectConfig) -> SourceScan:
    """Validate all configured roots, then list only their direct children."""
    roots = _configured_roots(config)
    issues: List[str] = []
    for index, root in enumerate(roots):
        for earlier_root in roots[:index]:
            if _paths_overlap(root.path, earlier_root.path):
                issues.append(
                    "Configured source roots overlap: {!s} and {!s}".format(
                        earlier_root.path, root.path
                    )
                )

    discovered: List[Tuple[SourceRoot, List[Tuple[Path, os.stat_result]]]] = []
    for root in roots:
        try:
            root_info = _assert_path_has_no_reparse_components(root.path)
            if root_info is None or not stat.S_ISDIR(root_info.st_mode):
                raise CorpusError("Source root is not a directory: {!s}".format(root.path))
            entries: List[Tuple[Path, os.stat_result]] = []
            with os.scandir(root.path) as iterator:
                for entry in iterator:
                    entry_path = Path(entry.path)
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        issues.append(
                            "Could not inspect source entry {!s}: {}".format(
                                entry_path, exc
                            )
                        )
                        continue

                    if _is_reparse_point(info):
                        issues.append(
                            "Symlink or junction is not allowed in flat source root: "
                            "{!s}".format(entry_path)
                        )
                    elif stat.S_ISDIR(info.st_mode):
                        issues.append(
                            "Nested directories are not allowed in flat source root: "
                            "{!s}".format(entry_path)
                        )
                    elif not stat.S_ISREG(info.st_mode):
                        issues.append(
                            "Non-regular filesystem entry is not allowed: {!s}".format(
                                entry_path
                            )
                        )
                    else:
                        entries.append((entry_path, info))
            discovered.append((root, entries))
        except CorpusError as exc:
            issues.append(str(exc))
        except OSError as exc:
            issues.append(
                "Could not inspect source root {!s}: {}".format(root.path, exc)
            )

    if issues:
        raise FlatFolderError(issues)

    files: List[SourceFile] = []
    unsupported: List[UnsupportedFile] = []
    for root, entries in discovered:
        for path, info in entries:
            extension = path.suffix.lower()
            file_format = extension[1:] if extension else "[no extension]"
            if extension in SUPPORTED_EXTENSIONS:
                files.append(
                    SourceFile(
                        path=path,
                        source_class=root.source_class,
                        file_format=file_format,
                        size_bytes=info.st_size,
                    )
                )
            else:
                unsupported.append(
                    UnsupportedFile(
                        path=path,
                        source_class=root.source_class,
                        file_format=file_format,
                        size_bytes=info.st_size,
                    )
                )

    sort_key = lambda item: (item.source_class, _normalized_path(item.path))
    return SourceScan(
        roots=roots,
        files=tuple(sorted(files, key=sort_key)),
        unsupported=tuple(sorted(unsupported, key=sort_key)),
    )


def _decode_bytes(
    raw: bytes, recover_text: bool = False
) -> Tuple[str, str, Tuple[str, ...]]:
    recovery_warnings: List[str] = []
    has_utf8_bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig" if has_utf8_bom else "utf-8", errors="strict")
        encoding = "utf-8-sig" if has_utf8_bom else "utf-8"
    except UnicodeDecodeError:
        try:
            text = raw.decode("windows-1252", errors="strict")
            encoding = "windows-1252"
        except UnicodeDecodeError as exc:
            if not recover_text:
                raise ExtractionError(
                    "Bytes are invalid as UTF-8 and Windows-1252; review the source "
                    "encoding."
                ) from exc
            undefined_bytes = frozenset((0x81, 0x8D, 0x8F, 0x90, 0x9D))
            removed = Counter(byte for byte in raw if byte in undefined_bytes)
            filtered = bytes(byte for byte in raw if byte not in undefined_bytes)
            text = filtered.decode("windows-1252", errors="strict")
            encoding = "windows-1252"
            recovery_warnings.append(
                "Removed {} undefined Windows-1252 byte(s): {}. "
                "Encoding fallback may be uncertain.".format(
                    sum(removed.values()),
                    ", ".join(
                        "0x{:02X}={}".format(byte, count)
                        for byte, count in sorted(removed.items())
                    ),
                )
            )

    suspicious = [
        character
        for character in text
        if character == "\ufffd"
        or (
            unicodedata.category(character) == "Cc"
            and character not in ("\n", "\r", "\t", "\f")
        )
    ]
    if suspicious:
        codes = sorted({ord(character) for character in suspicious})
        rendered = ", ".join("U+{:04X}".format(code) for code in codes[:5])
        if not recover_text:
            raise ExtractionError(
                "Decoded as {} but found suspicious control/replacement character(s): "
                "{}.".format(encoding, rendered),
                encoding=encoding,
            )
        removed_characters = Counter(suspicious)
        text = "".join(
            character for character in text if character not in removed_characters
        )
        recovery_warnings.append(
            "Removed {} suspicious control/replacement character(s): {}.".format(
                len(suspicious),
                ", ".join(
                    "U+{:04X}={}".format(ord(character), count)
                    for character, count in sorted(removed_characters.items())
                ),
            )
        )
    if recover_text and not text.strip():
        raise ExtractionError(
            "No usable text remains after decoding and character filtering.",
            encoding=encoding,
        )
    return text, encoding, tuple(recovery_warnings)


def _visible_html_soup(source: str) -> BeautifulSoup:
    soup = BeautifulSoup(source, "html.parser")
    hidden_classes = set()
    hidden_ids = set()
    hidden_rule = re.compile(
        r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE
    )
    simple_class = re.compile(r"\.([A-Za-z_][A-Za-z0-9_-]*)\Z")
    simple_id = re.compile(r"#([A-Za-z_][A-Za-z0-9_-]*)\Z")
    for style_tag in soup.find_all("style"):
        stylesheet = style_tag.get_text()
        for selectors, declarations in re.findall(
            r"([^{}]+)\{([^{}]*)\}", stylesheet
        ):
            if not hidden_rule.search(declarations):
                continue
            for selector in selectors.split(","):
                selector = selector.strip()
                class_match = simple_class.fullmatch(selector)
                id_match = simple_id.fullmatch(selector)
                if class_match:
                    hidden_classes.add(class_match.group(1))
                elif id_match:
                    hidden_ids.add(id_match.group(1))

    for class_name in hidden_classes:
        for tag in soup.find_all(class_=class_name):
            tag.decompose()
    for element_id in hidden_ids:
        for tag in soup.find_all(id=element_id):
            tag.decompose()
    for tag in soup.find_all(_REMOVED_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"hidden": True}):
        tag.decompose()
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()
    for tag in soup.find_all(style=True):
        if hidden_rule.search(tag.get("style", "")):
            tag.decompose()

    return soup


def _html_to_text(source: str) -> str:
    soup = _visible_html_soup(source)

    pieces: List[Tuple[str, bool]] = []

    def visit(node: object, in_preformatted: bool = False) -> None:
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            value = str(node)
            if not in_preformatted:
                value = re.sub(r"\s+", " ", value)
            if value:
                pieces.append((value, in_preformatted))
            return
        if not isinstance(node, Tag):
            return

        name = node.name.lower()
        if name == "br":
            pieces.append(("\n", False))
            return
        if name == "pre":
            pieces.append(("\n\n", False))
            for child in node.children:
                visit(child, in_preformatted=True)
            pieces.append(("\n\n", False))
            return

        is_block = name in _BLOCK_TAGS and not in_preformatted
        if is_block:
            pieces.append(("\n\n", False))
        for child in node.children:
            visit(child, in_preformatted)
        if is_block:
            pieces.append(("\n\n", False))

    if soup.body is not None:
        visit(soup.body)
    else:
        visit(soup)

    output: List[str] = []
    normal_text: List[str] = []

    def flush_normal_text() -> None:
        if not normal_text:
            return
        value = "".join(normal_text).replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[ \t]*\n[ \t]*", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        output.append(value.strip(" \n"))
        normal_text.clear()

    for value, is_preformatted in pieces:
        if is_preformatted:
            flush_normal_text()
            output.append(value.replace("\r\n", "\n").replace("\r", "\n"))
        else:
            normal_text.append(value)
    flush_normal_text()

    result = "".join(output)
    return result.strip("\n")


def _html_layout_spans(source: str, normalized_text: str) -> List[Dict[str, object]]:
    """Map meaningful HTML alignment/style blocks to normalized-text offsets."""
    soup = _visible_html_soup(source)
    spans: List[Dict[str, object]] = []

    def normalized_pattern(value: str) -> Optional[re.Pattern]:
        words = re.split(r"\s+", value.strip())
        if not words or not any(words):
            return None
        return re.compile(r"\s+".join(re.escape(word) for word in words if word))

    def add_tag_spans(role: str, tags: Sequence[Tag]) -> None:
        cursor = 0
        for tag in tags:
            value = _html_to_text(str(tag))
            pattern = normalized_pattern(value)
            if pattern is None:
                continue
            match = pattern.search(normalized_text, cursor)
            if match is None:
                continue
            span: Dict[str, object] = {
                "start_offset": match.start(),
                "end_offset": match.end(),
                "role": role,
                "tag": tag.name.lower(),
            }
            if tag.get("align"):
                span["align"] = str(tag.get("align")).casefold()
            if tag.get("face"):
                span["font_face"] = str(tag.get("face"))
            if tag.get("size"):
                span["font_size"] = str(tag.get("size"))
            if tag.get("bgcolor"):
                span["background"] = str(tag.get("bgcolor"))
            spans.append(span)
            cursor = match.end()

    center_candidates = [
        tag
        for tag in soup.find_all(True)
        if (
            tag.name.lower() == "center"
            and tag.find("table") is None
            and tag.find("center") is None
        )
        or (
            tag.name.lower() in _BLOCK_TAGS
            and (
                str(tag.get("align", "")).casefold() in ("center", "middle")
                or re.search(
                    r"text-align\s*:\s*center\b",
                    str(tag.get("style", "")),
                    re.IGNORECASE,
                )
            )
        )
    ]
    center_candidate_ids = {id(tag) for tag in center_candidates}
    centers = [
        tag
        for tag in center_candidates
        if not any(
            id(descendant) in center_candidate_ids
            for descendant in tag.find_all(True)
        )
    ]
    right_aligned = [
        tag
        for tag in soup.find_all(True)
        if str(tag.get("align", "")).casefold() == "right"
    ]
    styled_headings = [
        tag
        for tag in soup.find_all(True)
        if (
            tag.name.lower() in ("td", "font", "strong")
            and (
                tag.get("bgcolor") is not None
                or str(tag.get("size", "")) in ("+1", "+2", "+3")
            )
            and len(_html_to_text(str(tag)).strip()) <= 160
        )
    ]

    add_tag_spans("center", centers)
    add_tag_spans("right_aligned", right_aligned)
    add_tag_spans("styled_heading", styled_headings)
    for match in re.finditer(
        r"\bEND OF ACT\s+(?:ONE|TWO|THREE|FOUR|\d+)",
        normalized_text,
        re.IGNORECASE,
    ):
        spans.append(
            {
                "start_offset": match.start(),
                "end_offset": match.end(),
                "role": "act_marker",
                "tag": "text",
            }
        )
    spans.sort(key=lambda item: (item["start_offset"], item["end_offset"], item["role"]))
    return spans


def _read_source_bytes(path: Path) -> bytes:
    info = _assert_path_has_no_reparse_components(path)
    if info is None or not stat.S_ISREG(info.st_mode):
        raise CorpusError("Source is no longer a regular file: {!s}".format(path))
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CorpusError("Could not read source {!s}: {}".format(path, exc)) from exc


def _extract_bytes(
    raw: bytes, file_format: str
) -> Tuple[str, str, Tuple[str, ...]]:
    encoding, text, warnings, _ = _extract_bytes_with_layout(raw, file_format)
    return encoding, text, warnings


def _extract_bytes_with_layout(
    raw: bytes, file_format: str
) -> Tuple[str, str, Tuple[str, ...], Tuple[Dict[str, object], ...]]:
    text, encoding, recovery_warnings = _decode_bytes(
        raw, recover_text=file_format == "txt"
    )
    layout_spans: List[Dict[str, object]] = []
    if file_format in ("html", "htm"):
        source = text
        text = _html_to_text(source)
        layout_spans = _html_layout_spans(source, text)
    return encoding, text, recovery_warnings, tuple(layout_spans)


def _read_and_extract(
    path: Path, file_format: str
) -> Tuple[str, str, str, Tuple[str, ...]]:
    raw = _read_source_bytes(path)
    digest = hashlib.sha256(raw).hexdigest()
    encoding, text, recovery_warnings = _extract_bytes(raw, file_format)
    return digest, encoding, text, recovery_warnings


def _read_and_extract_with_layout(
    path: Path, file_format: str
) -> Tuple[str, str, str, Tuple[str, ...], Tuple[Dict[str, object], ...]]:
    """Extract normalized text plus layout spans for persistent import metadata."""
    raw = _read_source_bytes(path)
    digest = hashlib.sha256(raw).hexdigest()
    encoding, text, recovery_warnings, layout_spans = _extract_bytes_with_layout(
        raw, file_format
    )
    return digest, encoding, text, recovery_warnings, layout_spans


def _unsupported_as_dict(item: UnsupportedFile) -> Dict[str, object]:
    return {
        "filename": item.path.name,
        "path": str(item.path),
        "source_class": item.source_class,
        "format": item.file_format,
        "size_bytes": item.size_bytes,
        "status": "unsupported",
    }


def _validate_report_location(config: ProjectConfig, roots: Sequence[SourceRoot]) -> Path:
    storage = config.storage_directory
    for root in roots:
        if _paths_overlap(storage, root.path):
            raise CorpusError(
                "Storage directory overlaps a source folder; refusing to write "
                "the inventory report: {!s}".format(storage)
            )
    try:
        storage_info = _assert_path_has_no_reparse_components(
            storage, allow_missing_leaf=True
        )
    except CorpusError as exc:
        raise CorpusError("Unsafe storage directory: {}".format(exc)) from exc
    if storage_info is not None and not stat.S_ISDIR(storage_info.st_mode):
        raise CorpusError("Storage path is not a directory: {!s}".format(storage))
    return storage / "inventory_report.json"


def _write_report(path: Path, report: Dict[str, object]) -> None:
    temporary_path: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(path.parent),
            prefix=".inventory_report_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(report, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, path)
    except OSError as exc:
        cleanup_error = ""
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_error = " Temporary report cleanup also failed: {}".format(
                    cleanup_exc
                )
        raise CorpusError(
            "Could not write inventory report {!s}: {}{}".format(
                path, exc, cleanup_error
            )
        ) from exc


def create_inventory_report(config: ProjectConfig) -> Tuple[Path, Dict[str, object]]:
    """Inventory supported direct children and write a JSON report in storage."""
    scan = scan_sources(config)
    report_path = _validate_report_location(config, scan.roots)
    inventory_files: List[InventoryFile] = []

    for source_file in scan.files:
        digest = ""
        encoding: Optional[str] = None
        try:
            raw = _read_source_bytes(source_file.path)
            digest = hashlib.sha256(raw).hexdigest()
            encoding, text, recovery_warnings = _extract_bytes(
                raw, source_file.file_format
            )
            inventory_files.append(
                InventoryFile(
                    path=source_file.path,
                    source_class=source_file.source_class,
                    file_format=source_file.file_format,
                    size_bytes=len(raw),
                    sha256=digest,
                    status="ready",
                    encoding=encoding,
                    extracted_characters=len(text),
                    recovery_warnings=recovery_warnings,
                )
            )
        except CorpusError as exc:
            if isinstance(exc, ExtractionError):
                encoding = exc.encoding
            inventory_files.append(
                InventoryFile(
                    path=source_file.path,
                    source_class=source_file.source_class,
                    file_format=source_file.file_format,
                    size_bytes=source_file.size_bytes,
                    sha256=digest,
                    status="failed",
                    encoding=encoding,
                    error=str(exc),
                )
            )

    failed = sum(item.status == "failed" for item in inventory_files)
    report: Dict[str, object] = {
        "report_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": [
            {"path": str(root.path), "source_class": root.source_class}
            for root in scan.roots
        ],
        "summary": {
            "supported_files": len(inventory_files),
            "ready_files": len(inventory_files) - failed,
            "failed_files": failed,
            "recovered_files": sum(
                bool(item.recovery_warnings) for item in inventory_files
            ),
            "unsupported_files": len(scan.unsupported),
        },
        "files": [item.as_dict() for item in inventory_files],
        "unsupported": [_unsupported_as_dict(item) for item in scan.unsupported],
    }
    _write_report(report_path, report)
    return report_path, report


def preview_source_file(
    config: ProjectConfig,
    path: Path,
    recovery_warnings: Optional[List[str]] = None,
) -> Tuple[str, str, str, str]:
    """Extract one supported direct child from a configured source folder."""
    scan = scan_sources(config)
    requested_path = Path(os.path.abspath(os.fspath(path.expanduser())))
    requested_key = _normalized_path(requested_path)
    matches = [
        item
        for item in scan.files
        if _normalized_path(item.path) == requested_key
    ]
    if not matches:
        raise CorpusError(
            "Preview accepts only a direct-child .txt, .html, or .htm file from "
            "a configured source folder."
        )
    source_file = matches[0]
    digest, encoding, text, messages = _read_and_extract(
        source_file.path, source_file.file_format
    )
    if recovery_warnings is not None:
        recovery_warnings.extend(messages)
    else:
        for message in messages:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return source_file.source_class, encoding, digest, text
