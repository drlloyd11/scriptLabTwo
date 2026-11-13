import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scriptlab.config import ProjectConfig
from scriptlab.corpus import (
    CorpusError,
    FlatFolderError,
    _extract_bytes,
    _extract_bytes_with_layout,
    create_inventory_report,
    preview_source_file,
    scan_sources,
)


def make_config(
    base: Path,
    canonical_directories=None,
    fanfiction_directory=None,
    storage_directory=None,
) -> ProjectConfig:
    canonical = tuple(canonical_directories or (base / "scripts",))
    fanfiction = fanfiction_directory or base / "fanfiction"
    storage = storage_directory or base / "data"
    for directory in canonical + (fanfiction,):
        directory.mkdir(parents=True, exist_ok=True)
    return ProjectConfig(
        project_name="test",
        canonical_directories=canonical,
        fanfiction_directory=fanfiction,
        storage_directory=storage,
        embedding_model="test-model",
        ollama_url="http://127.0.0.1:11434",
    )


def read_report(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_inventory_records_hash_encoding_format_and_source_class(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "episode.txt"
    raw = b"\xef\xbb\xbfINT. ROOM - DAY\r\n\nMARA\r\nHello."
    source.write_bytes(raw)

    report_path, report = create_inventory_report(config)

    item = report["files"][0]
    assert report_path == config.storage_directory / "inventory_report.json"
    assert item["filename"] == "episode.txt"
    assert item["source_class"] == "canonical"
    assert item["format"] == "txt"
    assert item["size_bytes"] == len(raw)
    assert item["sha256"] == hashlib.sha256(raw).hexdigest()
    assert item["encoding"] == "utf-8-sig"
    assert item["status"] == "ready"
    assert item["extracted_characters"] == len("INT. ROOM - DAY\r\n\nMARA\r\nHello.")
    assert read_report(report_path) == report


def test_windows_1252_text_is_decoded_strictly_and_recorded(tmp_path):
    config = make_config(tmp_path)
    source = config.fanfiction_directory / "story.txt"
    source.write_bytes("caf\u00e9".encode("windows-1252"))

    _, report = create_inventory_report(config)

    item = report["files"][0]
    assert item["source_class"] == "fanfiction"
    assert item["encoding"] == "windows-1252"
    assert item["status"] == "ready"
    assert item["extracted_characters"] == 4


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\x81", "No usable text remains"),
        (b"\x00\x01", "No usable text remains"),
    ],
)
def test_suspicious_or_undecodable_text_is_reported_as_failed(
    tmp_path, raw, message
):
    config = make_config(tmp_path)
    (config.canonical_directories[0] / "damaged.txt").write_bytes(raw)

    _, report = create_inventory_report(config)

    item = report["files"][0]
    assert item["status"] == "failed"
    assert item["sha256"] == hashlib.sha256(raw).hexdigest()
    assert message in item["error"]
    assert report["summary"]["failed_files"] == 1


@pytest.mark.parametrize(
    ("raw", "expected", "warning"),
    [
        (b"MARA\x00\nHello.\r\n\tNext.\f", "MARA\nHello.\r\n\tNext.\f", "U+0000=1"),
        (b"Keep\xef\xbf\xbdtext", "Keeptext", "U+FFFD=1"),
        (b"caf\xe9\x81\x00\nNext.", "caf\u00e9\nNext.", "0x81=1"),
    ],
)
def test_txt_recovery_preserves_remaining_text_and_records_removals(
    tmp_path, raw, expected, warning
):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "recover.txt"
    source.write_bytes(raw)

    _, report = create_inventory_report(config)
    messages = []
    _, _, digest, text = preview_source_file(config, source, messages)

    assert text == expected
    assert source.read_bytes() == raw
    assert digest == hashlib.sha256(raw).hexdigest()
    assert report["files"][0]["status"] == "ready"
    assert report["files"][0]["extracted_characters"] == len(expected)
    assert report["summary"]["recovered_files"] == 1
    assert report["summary"]["failed_files"] == 0
    assert messages == report["files"][0]["recovery_warnings"]
    assert warning in " ".join(messages)


def test_html_does_not_silently_apply_txt_recovery(tmp_path):
    config = make_config(tmp_path)
    (config.canonical_directories[0] / "damaged.htm").write_bytes(b"<p>hello\x00</p>")

    _, report = create_inventory_report(config)

    assert report["files"][0]["status"] == "failed"
    assert report["summary"]["recovered_files"] == 0


def test_html_extracts_visible_text_and_preserves_layout(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "episode.htm"
    source.write_text(
        """<!doctype html>
        <html>        <head><title>Ignored title</title><style>.hidden-rule { display:none }</style></head>
        <body>
          <nav>Ignored navigation</nav>
          <script>ignored_script()</script>
          <!-- Ignore this source comment. -->
          <div class="hidden-rule">Hidden by CSS class</div>
          <div style="display: none">Hidden by inline style</div>
          <h1>Episode heading</h1>
          <p>INT. ROOM - DAY<br>MARA<br>Hello &amp; welcome.</p>
          <pre>JON
  I kept the original spacing.

MARA
  Me too.</pre>
          <div hidden>Hidden text</div>
          <div aria-hidden="true">Screen reader hidden</div>
        </body></html>""",
        encoding="utf-8",
    )

    source_class, encoding, _, text = preview_source_file(config, source)

    assert source_class == "canonical"
    assert encoding == "utf-8"
    assert "Episode heading" in text
    assert "INT. ROOM - DAY\nMARA\nHello & welcome." in text
    assert "JON\n  I kept the original spacing.\n\nMARA\n  Me too." in text
    for excluded in (
        "Ignored title",
        "Ignored navigation",
        "ignored_script",
        "Ignore this source comment.",
        "display:none",
        "Hidden by CSS class",
        "Hidden by inline style",
        "Hidden text",
        "Screen reader hidden",
    ):
        assert excluded not in text


def test_html_layout_spans_map_to_unchanged_normalized_text():
    raw = (
        '<html><body><center><font face="Verdana">XANDER<br>We get out.'
        '</font></center><p>Willow grabs a bag.</p>'
        '<p align="center" style="text-align:center"><span '
        'style="font-size:10.0pt;font-family:Verdana">ANGEL<br>Stay back.'
        '</span></p>'
        '<div align="right">CUT TO:</div>'
        '<table><tr><td bgcolor="#6666ff"><font size="+2">Act Two</font>'
        '</td></tr></table><center>END OF ACT TWO</center></body></html>'
    ).encode("utf-8")

    encoding, text, warnings = _extract_bytes(raw, "html")
    layout_encoding, layout_text, layout_warnings, spans = _extract_bytes_with_layout(
        raw, "html"
    )
    rendered = {
        span["role"]: layout_text[span["start_offset"] : span["end_offset"]]
        for span in spans
    }

    assert encoding == layout_encoding == "utf-8"
    assert text == layout_text
    assert warnings == layout_warnings == ()
    assert rendered["center"] == "END OF ACT TWO"
    assert "XANDER\nWe get out." in [
        layout_text[span["start_offset"] : span["end_offset"]]
        for span in spans
        if span["role"] == "center"
    ]
    assert "ANGEL\nStay back." in [
        layout_text[span["start_offset"] : span["end_offset"]]
        for span in spans
        if span["role"] == "center"
    ]
    assert rendered["right_aligned"] == "CUT TO:"
    assert rendered["styled_heading"] == "Act Two"
    assert rendered["act_marker"] == "END OF ACT TWO"


def test_html_div_and_paragraph_boundaries_are_separated(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "layout.html"
    source.write_text(
        "<main><p>First paragraph.</p><div>Second block.</div>"
        "<p>Third paragraph.</p></main>",
        encoding="utf-8",
    )

    _, _, _, text = preview_source_file(config, source)

    assert "First paragraph.\n\nSecond block.\n\nThird paragraph." in text


def test_preformatted_html_preserves_indentation_and_blank_lines(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "pre.html"
    source.write_text(
        "<html><body><pre>    first line\n\n\nsecond line\n    final  \n</pre>"
        "</body></html>",
        encoding="utf-8",
    )

    _, _, _, text = preview_source_file(config, source)

    assert "    first line\n\n\nsecond line\n    final  " in text


def test_preflight_rejects_nested_directory_before_reading_or_writing(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    (config.canonical_directories[0] / "nested").mkdir()
    (config.canonical_directories[0] / "episode.txt").write_text(
        "MARA\nHello.", encoding="utf-8"
    )

    def fail_if_source_is_read(*args, **kwargs):
        raise AssertionError("source read happened before the flat-folder check")

    monkeypatch.setattr(Path, "read_bytes", fail_if_source_is_read)

    with pytest.raises(FlatFolderError, match="Nested directories"):
        create_inventory_report(config)

    assert not config.storage_directory.exists()


def test_preflight_rejects_overlapping_source_roots(tmp_path):
    canonical = tmp_path / "sources"
    fanfiction = canonical / "fanfiction"
    config = make_config(
        tmp_path,
        canonical_directories=(canonical,),
        fanfiction_directory=fanfiction,
    )

    with pytest.raises(FlatFolderError, match="overlap"):
        scan_sources(config)


def test_unsupported_files_are_reported_without_hashing(tmp_path):
    config = make_config(tmp_path)
    (config.fanfiction_directory / "notes.pdf").write_bytes(b"not opened")

    _, report = create_inventory_report(config)

    assert report["summary"]["unsupported_files"] == 1
    assert report["unsupported"][0]["filename"] == "notes.pdf"
    assert report["unsupported"][0]["source_class"] == "fanfiction"
    assert "sha256" not in report["unsupported"][0]


def test_preview_rejects_files_outside_configured_directories(tmp_path):
    config = make_config(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("not a configured source", encoding="utf-8")

    with pytest.raises(CorpusError, match="direct-child"):
        preview_source_file(config, outside)


def test_inventory_does_not_follow_child_symlink(tmp_path):
    config = make_config(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside source", encoding="utf-8")
    link = config.canonical_directories[0] / "linked.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("Symlink creation is unavailable: {}".format(exc))

    with pytest.raises(FlatFolderError, match="Symlink or junction"):
        create_inventory_report(config)

    assert not config.storage_directory.exists()


def test_preflight_rejects_reparse_point_entry(tmp_path, monkeypatch):
    import scriptlab.corpus as corpus

    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "reparse.txt"
    source.write_text("not opened", encoding="utf-8")
    real_scandir = os.scandir

    class FakeEntry:
        def __init__(self, entry):
            self._entry = entry
            self.name = entry.name
            self.path = entry.path

        def stat(self, follow_symlinks=True):
            if Path(self.path) == source:
                return SimpleNamespace(
                    st_mode=stat.S_IFLNK,
                    st_file_attributes=0x0400,
                    st_size=0,
                )
            return self._entry.stat(follow_symlinks=follow_symlinks)

    class FakeScandir:
        def __init__(self, path):
            self._iterator = real_scandir(path)

        def __enter__(self):
            self._iterator.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return self._iterator.__exit__(exc_type, exc_value, traceback)

        def __iter__(self):
            return (FakeEntry(entry) for entry in self._iterator)

    monkeypatch.setattr(corpus.os, "scandir", FakeScandir)

    with pytest.raises(FlatFolderError, match="Symlink or junction"):
        scan_sources(config)

    assert not config.storage_directory.exists()
