"""Conservative, deterministic parsing of ScriptLab documents."""

from dataclasses import dataclass
from bisect import bisect_right
import hashlib
import json
import re
from typing import Dict, List, Optional, Sequence, Tuple


PARSER_VERSION = "scriptlab-parser-v18"
CHUNKER_VERSION = "scriptlab-chunker-v16"
COMPOSITE_SEPARATOR = "\n[...]\n"
_MAX_DIALOGUE_CHUNK_CHARS = 2200

_SCREENPLAY_HEADING = re.compile(
    r"^(?:INT\.|EXT\.|INT/EXT\.|EXT/INT\.|I/E\.|EST\.)\s+.+$", re.IGNORECASE
)
_UNCERTAIN_SCREENPLAY_HEADING = re.compile(r"^(?:INT|EXT)\b.*\S$", re.IGNORECASE)
_UNCERTAIN_SLUG = re.compile(
    r"^[A-Z0-9][A-Z0-9'\u2019 .()-]*\s-\s.*\s-\s"
    r"(?:DAY|NIGHT|DAWN|DUSK|CONTINUOUS|LATER|MOMENTS LATER)$"
)
_PROSE_HEADING = re.compile(
    r"^(?:chapter\s+(?:\d+|[ivxlcdm]+|[a-z][a-z0-9 '\u2019-]*)|"
    r"\*{3,}|-{3,}|_{3,}|~{3,})$",
    re.IGNORECASE,
)
_TRAILING_CUE = re.compile(r" {2,}([A-Z][A-Z0-9'\u2019 .(),#-]{0,43})$")
_ACTION_LABELS = frozenset(
    (
        "ANGLE",
        "BLACKOUT",
        "CREDITS",
        "CUT TO",
        "FADE IN",
        "FADE OUT",
        "FADE TO BLACK",
        "END OF TEASER",
        "END OF ACT",
        "END OF EPISODE",
        "END OF SHOW",
        "CONTINUED",
        "MAIN TITLES",
        "BLACK OUT",
        "BLACK",
        "STAT",
        "OMITTED",
        "RESUME",
        "TO BE CONTINUED",
        "THE END",
    )
)
_NON_SPEAKER_LABELS = frozenset(("I",))
_NON_AMBIGUOUS_SPEAKER_LABELS = frozenset(("M. C.'S VOICE", "M.C.'S VOICE"))
_USER_STAGE_DIRECTION_TERMS = re.compile(
    r"\b(?:CLOSES|VIEW|WIDEN\w*|END OF PART|OMITTED|RESUME|"
    r"P\.?O\.?V\.?|POINT OF VIEW)\b",
    re.IGNORECASE,
)
_SHOT_OR_SLUG_LABEL = re.compile(
    r"\s-\s.*\b(?:DAY|NIGHT|MORNING|EVENING|AFTERNOON|DAWN|DUSK|SUNSET|SUNRISE|LATER|"
    r"CONTINUOUS|CONTINUING)$|"
    r"\bP\.?O\.?V\b|"
    r"^(?:ON|BACK ON|CLOSE ON|CLOSER ON|WIDE ON|ANGLE ON|PUSH IN ON|FAVORING)\s+\S|"
    r"^END (?:MONTAGE|OF SEASON)$"
)
_ACTION_DESCRIPTION_LABEL = re.compile(
    r"\b(?:IS|ARE|WAS|WERE)\s+STANDING\b|\bFLIES THROUGH\b|\bWINCES?\b"
)
_ACTION_DIRECTION_BODY = re.compile(r"^(?:react(?:s)?|wince(?:s)?)\b", re.IGNORECASE)
_ACTION_PARAGRAPH_START = re.compile(
    r"^(?:A|AN|THE|HE|SHE|THEY|IT|AS|MEANWHILE|SUDDENLY)\b", re.IGNORECASE
)
_OBJECT_INSERT_NOUNS = frozenset(
    ("BAG", "BODY", "CAR", "DOOR", "GUN", "HAND", "HEAD", "LETTER", "MIRROR", "PHONE", "RADIO", "SWORD", "WINDOW")
)
_OBJECT_INSERT_ACTION = re.compile(
    r"^(?:falls?|drops?|hits?|lands?|opens?|closes?|rolls?|slides?|spins?|"
    r"raises?|lowers?)\b",
    re.IGNORECASE,
)
_NAMED_ACTION_PARAGRAPH = re.compile(
    r"^[A-Z][a-z]+(?:['\u2019-][A-Z]?[a-z]+)?\s+"
    r"(?:rises?|steps?|moves?|turns?|looks?|starts?|begins?|walks?|runs?|"
    r"grabs?|takes?|pulls?|pushes?|reaches?|heads?|dives?|falls?|drops?|"
    r"spins?|backs?|comes?|laughs?|smiles?|nods?|shrugs?|scrambles?|"
    r"stands?|sits?|opens?|closes?|stumbles?|dashes?|leaps?|scoots?|leans?|"
    r"hurries?|wheels?|bolts?|catches?|raises?|lowers?|shakes?)\b",
    re.IGNORECASE,
)
_GROUP_ACTION_START = re.compile(
    r"^(?:FLANK|MOVE|TURN|FOLLOW|LOOK|RUN|WALK|GRAB|HOLD|PUSH|PULL|OPEN|"
    r"CLOSE|STEP|STAND|SIT|ENTER|EXIT|ATTACK|DUCK|SWING|KICK|STAGGER|"
    r"CHASE|CATCH|LEAVE|RETURN|APPROACH|REACH|NOD|SMILE|SHRUG)\b",
    re.IGNORECASE,
)
_ACTION_STARTS = frozenset(
    ("A", "AN", "THE", "HE", "SHE", "WE", "THEY", "IT", "THIS", "THAT", "THERE", "HERE")
)
_ACTION_PHRASE_STARTS = frozenset(
    ("WE", "PULL", "PUSH", "TRACK", "CAMERA", "ZOOM", "PAN", "DOLLY", "MOVE", "MOVES")
)
camera_directions = frozenset(
    (
        "ANGLE",
        "ANGLE -",
        "A REALLY LONG LENS SHOT -",
        "DRAMATIC ANGLE -",
        "BACK TO",
        "BACK TO SCENE",
        "CLOSE -",
        "CLOSE SHOTS",
        "CLOSE UP",
        "CLOSE-UP",
        "CUT TO",
        "FADE IN",
        "FADE OUT",
        "FLASHBACK",
        "PULL BACK",
        "PUSH IN",
        "PUSHING ONTO",
        "REVERSE",
        "WE PAN",
        "WE SEE",
        "WIDER",
        "WIDER ANGLE",
    )
)
_CAMERA_FOCUS_NOUNS = frozenset(
    ("CAR", "COVER", "EYE", "EYES", "FACE", "FIGURE", "HAND", "KILLER", "MOON", "THUMB")
)
_VISUAL_DETERMINERS = frozenset(("A", "AN", "HER", "HIS", "ITS", "THE", "THIS", "THAT"))
_SONG_CUE = re.compile(
    r"\b(?:begins? to sing|starts? to sing|begins? singing|starts? singing|"
    r"takes? the song|sings\b|singing\b|orchestration backing)\b",
    re.IGNORECASE,
)
_SONG_END = re.compile(
    r"\b(?:the number is over|the number ends|the song is over|the song ends)\b",
    re.IGNORECASE,
)
_GROUP_SPEAKER_LABELS = frozenset(
    ("GROUP", "ALL", "CHOIR", "ENSEMBLE", "DEMON AND VAMPS")
)


@dataclass(frozen=True)
class SourceSpan:
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class ParsedScene:
    scene_id: str
    scene_ordinal: int
    scene_type: str
    scene_heading: Optional[str]
    start_offset: int
    end_offset: int
    scene_text: str


@dataclass(frozen=True)
class ParsedDialogueTurn:
    turn_id: str
    scene_id: Optional[str]
    turn_ordinal: int
    start_offset: int
    end_offset: int
    turn_text: str
    original_speaker_label: str
    attribution_status: str
    speaker_character_id: Optional[str] = None


@dataclass(frozen=True)
class ParsedChunk:
    chunk_id: str
    chunker_version: str
    chunk_ordinal: int
    chunk_type: str
    spans: Tuple[SourceSpan, ...]
    chunk_text: str


@dataclass(frozen=True)
class ParseWarning:
    warning_id: str
    scene_id: Optional[str]
    start_offset: int
    end_offset: int
    warning_code: str
    message: str
    parser_version: str


@dataclass(frozen=True)
class ParseResult:
    document_id: str
    parser_version: str
    scenes: Tuple[ParsedScene, ...]
    dialogue_turns: Tuple[ParsedDialogueTurn, ...]
    chunks: Tuple[ParsedChunk, ...]
    warnings: Tuple[ParseWarning, ...]


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _Candidate:
    start: int
    label_end: int
    label: str
    ambiguous: bool


def _stable_id(prefix: str, payload: object) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "{}_{}".format(prefix, digest)


def _split_lines(text: str) -> List[_Line]:
    lines: List[_Line] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        content = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        lines.append(_Line(offset, offset + len(content), content))
        offset += len(raw_line)
    if text and (not lines or offset < len(text)):
        lines.append(_Line(offset, len(text), text[offset:]))
    return lines


def _paragraphs(lines: Sequence[_Line]) -> List[Tuple[int, int]]:
    paragraphs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, line in enumerate(lines):
        if line.text.strip():
            if start is None:
                start = index
        elif start is not None:
            paragraphs.append((start, index - 1))
            start = None
    if start is not None:
        paragraphs.append((start, len(lines) - 1))
    return paragraphs


def _scene_anchors(lines: Sequence[_Line]) -> Tuple[str, List[Tuple[int, str]]]:
    screenplay = [
        (index, line.text.strip())
        for index, line in enumerate(lines)
        if _is_scene_heading(line.text)
    ]
    if screenplay:
        return "screenplay", screenplay
    prose = [
        (index, line.text.strip())
        for index, line in enumerate(lines)
        if _PROSE_HEADING.match(line.text.strip())
    ]
    return ("prose", prose) if prose else ("prose", [])


def _make_scenes(
    document_id: str,
    text: str,
    lines: Sequence[_Line],
    parser_version: str,
) -> Tuple[ParsedScene, ...]:
    scene_type, anchors = _scene_anchors(lines)
    scenes: List[ParsedScene] = []
    for ordinal, (line_index, heading) in enumerate(anchors):
        start = lines[line_index].start
        end = lines[anchors[ordinal + 1][0]].start if ordinal + 1 < len(anchors) else len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            continue
        scenes.append(
            ParsedScene(
                scene_id=_stable_id(
                    "scene",
                    [document_id, parser_version, scene_type, ordinal, start, end],
                ),
                scene_ordinal=len(scenes),
                scene_type=scene_type,
                scene_heading=heading,
                start_offset=start,
                end_offset=end,
                scene_text=text[start:end],
            )
        )
    return tuple(scenes)


def _label_parts(label: str) -> Tuple[str, bool]:
    cleaned = label.strip()
    if (
        not cleaned
        or cleaned != cleaned.upper()
        or not re.fullmatch(r"[A-Z0-9'\u2019 .(),#/-]+", cleaned)
    ):
        return "", False
    action_label = cleaned.rstrip(".")
    normalized_label = re.sub(r"\s+", " ", action_label)
    if normalized_label in _NON_SPEAKER_LABELS or normalized_label.endswith("--"):
        return "", False
    if _USER_STAGE_DIRECTION_TERMS.search(normalized_label):
        return "", False
    if _SHOT_OR_SLUG_LABEL.search(normalized_label) or _ACTION_DESCRIPTION_LABEL.search(
        normalized_label
    ):
        return "", False
    if _is_camera_direction(normalized_label) or action_label in _ACTION_LABELS or re.match(
        r"^(?:END OF ACT\b|ACT (?:ONE|TWO|THREE|FOUR|\d+)\b|"
        r"SAME SCENE\b|ANGLE\b|CLOSE UP\b|WIDER ANGLE\b)",
        action_label,
    ):
        return "", False
    base_label = re.sub(r"\s+\([^()]*\)$", "", cleaned)
    base_label = re.sub(
        r"\s+(?:VOICE ON MACHINE|ON MACHINE|ON TAPE|ON PHONE|OVER PHONE|V\.O\.)$",
        "",
        base_label,
    )
    tokens = re.findall(r"#[ ]?\d+|[A-Z]+(?:['\u2019][A-Z]+)?", base_label)
    if not tokens:
        return "", False
    has_numbered_cue = "#" in cleaned
    ambiguous = len(tokens) > 3 or (
        len(tokens) == 3 and tokens[0] in _ACTION_STARTS
    ) or (
        len(tokens) >= 2 and tokens[0] in _ACTION_PHRASE_STARTS
    )
    return (
        cleaned,
        ambiguous
        and not has_numbered_cue
        and cleaned not in _NON_AMBIGUOUS_SPEAKER_LABELS,
    )


def _is_camera_direction(label: str) -> bool:
    normalized = label.replace("\u2013", "-").replace("\u2014", "-").strip()
    if normalized in camera_directions or any(
        normalized.startswith(direction)
        for direction in camera_directions
        if direction.endswith(" -")
    ):
        return True
    words = re.findall(r"#[ ]?\d+|[A-Z]+(?:['\u2019][A-Z]+)?", normalized)
    if not any(word in _CAMERA_FOCUS_NOUNS for word in words):
        return False
    return bool(
        words
        and words[0] in _VISUAL_DETERMINERS
        or re.search(r"(?:'S|’S)(?:\s|$)", label)
    )


def _is_action_paragraph(paragraph: str, speaker_label: str) -> bool:
    stripped = paragraph.strip()
    if not stripped or re.fullmatch(r"\([^()]*\)", stripped):
        return False
    normalized = stripped.replace("\u2013", "-").replace("\u2014", "-")
    if (
        _is_scene_heading(normalized)
        or _UNCERTAIN_SLUG.match(normalized)
        or _is_camera_direction(normalized)
        or _ACTION_DIRECTION_BODY.match(normalized)
        or _ACTION_PARAGRAPH_START.match(normalized)
        or _NAMED_ACTION_PARAGRAPH.match(normalized)
        or re.match(r"^[A-Z][a-z]+\s+(?:begins?|starts?|continues?)\b", normalized)
    ):
        return True
    label_words = re.findall(r"[A-Z]+", speaker_label.upper())
    if (
        1 < len(label_words) <= 4
        and label_words[-1] in _OBJECT_INSERT_NOUNS
        and _OBJECT_INSERT_ACTION.match(normalized)
    ):
        return True
    is_group_label = bool(re.search(r"(?:\bAND\b|/)", speaker_label))
    return is_group_label and bool(_GROUP_ACTION_START.match(normalized))


def _dialogue_action_boundary(
    text: str, start: int, end: int, speaker_label: str
) -> int:
    body = text[start:end]
    separators = list(re.finditer(r"\n[ \t]*\n+", body))
    is_group_label = bool(re.search(r"(?:\bAND\b|/)", speaker_label))
    first_paragraph = next(
        (
            paragraph
            for paragraph in re.split(r"\n[ \t]*\n+", body)
            if paragraph.strip()
        ),
        "",
    )
    label_words = re.findall(r"[A-Z]+", speaker_label.upper())
    if (
        1 < len(label_words) <= 4
        and label_words[-1] in _OBJECT_INSERT_NOUNS
        and _OBJECT_INSERT_ACTION.match(first_paragraph.strip())
    ):
        return start
    if is_group_label and _is_action_paragraph(first_paragraph, speaker_label):
        return start
    speech_seen = False
    for index, separator in enumerate(separators):
        previous_start = separators[index - 1].end() if index else 0
        previous_paragraph = body[previous_start : separator.start()].strip()
        if previous_paragraph and not re.fullmatch(r"\([^()]*\)", previous_paragraph):
            speech_seen = True
        if not speech_seen:
            continue
        next_start = separator.end()
        next_end = separators[index + 1].start() if index + 1 < len(separators) else len(body)
        next_paragraph = body[next_start:next_end]
        if _is_action_paragraph(next_paragraph, speaker_label):
            return start + separator.start()
    return end


def _is_scene_heading(label: str) -> bool:
    normalized = label.strip()
    if _is_camera_direction(normalized):
        return False
    return bool(
        _SCREENPLAY_HEADING.match(normalized)
        or _UNCERTAIN_SCREENPLAY_HEADING.match(normalized)
        or _UNCERTAIN_SLUG.match(normalized)
    )


def _line_candidate(line: _Line) -> Optional[_Candidate]:
    stripped = line.text.strip()
    if _is_scene_heading(stripped) or re.fullmatch(r"\([^()]*\)", stripped):
        return None
    leading = len(line.text) - len(line.text.lstrip())
    label, ambiguous = _label_parts(stripped)
    if label and label == stripped and line.text.strip().upper() == line.text.strip():
        if "INT." not in label and "EXT." not in label:
            return _Candidate(line.start + leading, line.end, label, ambiguous)

    match = _TRAILING_CUE.search(line.text)
    if match:
        candidate = match.group(1).strip()
        label, ambiguous = _label_parts(candidate)
        if label == candidate:
            start = line.start + match.start(1)
            return _Candidate(start, start + len(candidate), candidate, ambiguous)
    return None


def _scene_for_offset(scenes: Sequence[ParsedScene], offset: int) -> Optional[str]:
    for scene in scenes:
        if scene.start_offset <= offset < scene.end_offset:
            return scene.scene_id
    return None


def _make_warning(
    document_id: str,
    parser_version: str,
    scene_id: Optional[str],
    start: int,
    end: int,
    code: str,
    message: str,
) -> ParseWarning:
    return ParseWarning(
        warning_id=_stable_id(
            "warning", [document_id, parser_version, code, start, end, message]
        ),
        scene_id=scene_id,
        start_offset=start,
        end_offset=end,
        warning_code=code,
        message=message,
        parser_version=parser_version,
    )


def _song_ranges(
    text: str, lines: Sequence[_Line], scenes: Sequence[ParsedScene]
) -> Tuple[Tuple[int, int], ...]:
    ranges: List[Tuple[int, int]] = []
    for line in lines:
        if not line.text.strip() or line.text.strip() == line.text.strip().upper():
            continue
        if not _SONG_CUE.search(line.text):
            continue
        scene_id = _scene_for_offset(scenes, line.start)
        scene_end = next(
            (scene.end_offset for scene in scenes if scene.scene_id == scene_id),
            len(text),
        )
        song_end = _SONG_END.search(text, line.end, scene_end)
        end = song_end.start() if song_end else scene_end
        if end > line.start:
            ranges.append((line.start, end))

    merged: List[Tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _extend_song_ranges(
    text: str,
    candidates: Sequence[_Candidate],
    scenes: Sequence[ParsedScene],
    song_ranges: Sequence[Tuple[int, int]],
    known_speakers: set,
) -> Tuple[Tuple[int, int], ...]:
    expanded: List[Tuple[int, int]] = []
    for range_start, range_end in song_ranges:
        scene_id = _scene_for_offset(scenes, range_start)
        scene = next(
            (item for item in scenes if item.scene_id == scene_id),
            None,
        )
        if scene is None:
            expanded.append((range_start, range_end))
            continue

        starts = []
        for candidate_index, candidate in enumerate(candidates):
            if not (scene.start_offset <= candidate.start < range_start):
                continue
            if candidate.ambiguous or candidate.label not in known_speakers:
                continue
            next_candidate = (
                candidates[candidate_index + 1].start
                if candidate_index + 1 < len(candidates)
                else scene.end_offset
            )
            if next_candidate > scene.end_offset:
                next_candidate = scene.end_offset
            cue_body = text[candidate.label_end:next_candidate]
            if any(character.islower() for character in cue_body):
                continue
            lyric_candidates = sum(
                item.ambiguous
                for item in candidates[candidate_index + 1 :]
                if candidate.start < item.start < range_start
                and item.start < scene.end_offset
            )
            if lyric_candidates >= 3:
                starts.append(candidate.start)
        expanded.append((min(starts) if starts else range_start, range_end))

    merged: List[Tuple[int, int]] = []
    for start, end in sorted(expanded):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _offset_in_ranges(offset: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    range_index = bisect_right([start for start, _ in ranges], offset) - 1
    return range_index >= 0 and offset < ranges[range_index][1]


def _layout_roles_at(offset: int, layout_spans: Sequence[Dict[str, object]]) -> set:
    return {
        str(span["role"])
        for span in layout_spans
        if int(span["start_offset"]) <= offset < int(span["end_offset"])
    }


def _is_known_speaker_label(label: str, known_speakers: set) -> bool:
    if label in known_speakers or label in _GROUP_SPEAKER_LABELS:
        return True
    if re.search(r"\s+\((?:O\.S\.|V\.O\.|CONT'D|CONTINUED)\)$", label):
        return True
    combined_names = re.split(r"\s*/\s*|\s+AND\s+", label)
    return len(combined_names) > 1 and all(
        name.strip() in known_speakers for name in combined_names
    )


def _parse_dialogue(
    document_id: str,
    source_class: str,
    text: str,
    lines: Sequence[_Line],
    scenes: Sequence[ParsedScene],
    parser_version: str,
    layout_spans: Sequence[Dict[str, object]] = (),
) -> Tuple[Tuple[ParsedDialogueTurn, ...], Tuple[ParseWarning, ...]]:
    if source_class != "canonical":
        return (), ()

    candidates: List[_Candidate] = []
    layout_boundaries: List[int] = []
    has_center_layout = any(span.get("role") == "center" for span in layout_spans)

    def candidate_has_dialogue_layout(offset: int) -> bool:
        roles = _layout_roles_at(offset, layout_spans)
        if "right_aligned" in roles:
            return False
        return not has_center_layout or "center" in roles

    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        split_voiceover = re.fullmatch(
            r"\s*([A-Z][A-Z0-9'\u2019 .-]*)\s+\(V\.", line.text
        )
        if (
            split_voiceover is not None
            and line_index + 1 < len(lines)
            and re.fullmatch(r"\s*O\.\)\s*", lines[line_index + 1].text)
        ):
            speaker = split_voiceover.group(1).strip()
            start = line.start + len(line.text) - len(line.text.lstrip())
            if candidate_has_dialogue_layout(start):
                candidates.append(
                    _Candidate(
                        start,
                        lines[line_index + 1].end,
                        "{} (V.O.)".format(speaker),
                        False,
                    )
                )
            else:
                layout_boundaries.append(start)
            line_index += 2
            continue
        candidate = _line_candidate(line)
        if candidate is not None:
            if candidate_has_dialogue_layout(candidate.start):
                candidates.append(candidate)
            else:
                layout_boundaries.append(candidate.start)
        line_index += 1
    for span in layout_spans:
        start = int(span["start_offset"])
        end = int(span["end_offset"])
        role = span.get("role")
        if role in ("act_marker", "right_aligned"):
            layout_boundaries.append(start)
        elif role == "styled_heading":
            heading = text[start:end].strip()
            if re.fullmatch(
                r"(?:ACT\s+(?:ONE|TWO|THREE|FOUR|\d+)|TEASER|TAG)",
                heading,
                re.IGNORECASE,
            ):
                layout_boundaries.append(start)
    scene_boundaries = sorted(
        {
            scene.start_offset for scene in scenes
        }.union(
            line.start
            for line in lines
            if _UNCERTAIN_SCREENPLAY_HEADING.match(line.text.strip())
            or _UNCERTAIN_SLUG.match(line.text.strip())
        ).union(layout_boundaries)
    )
    song_ranges = _song_ranges(text, lines, scenes)
    known_speakers = set(_GROUP_SPEAKER_LABELS)
    for index, candidate in enumerate(candidates):
        if candidate.ambiguous or _offset_in_ranges(candidate.start, song_ranges):
            continue
        next_candidate = (
            candidates[index + 1].start
            if index + 1 < len(candidates)
            else len(text)
        )
        scene_index = bisect_right(scene_boundaries, candidate.start)
        next_scene = (
            scene_boundaries[scene_index]
            if scene_index < len(scene_boundaries)
            else len(text)
        )
        candidate_end = min(next_candidate, next_scene)
        candidate_end = _dialogue_action_boundary(
            text, candidate.label_end, candidate_end, candidate.label
        )
        body = text[candidate.label_end:candidate_end]
        if _ACTION_DIRECTION_BODY.match(body.strip()):
            continue
        if any(character.islower() for character in body):
            known_speakers.add(candidate.label)
    for line in lines:
        if not _SONG_CUE.search(line.text):
            continue
        named_speaker = re.search(r"\b([A-Z][A-Z0-9']*)\s+sings\b", line.text)
        if named_speaker is None:
            named_speaker = re.search(
                r"\b([A-Z][A-Z0-9']*)\s+takes? the song\b", line.text, re.IGNORECASE
            )
        if named_speaker and named_speaker.group(1) not in ("HE", "SHE", "THEY"):
            known_speakers.add(named_speaker.group(1).upper())
    song_ranges = _extend_song_ranges(
        text, candidates, scenes, song_ranges, known_speakers
    )

    dialogue_candidates: List[_Candidate] = []
    active_speakers: Dict[Optional[str], str] = {}
    for candidate_index, candidate in enumerate(candidates):
        next_candidate = (
            candidates[candidate_index + 1].start
            if candidate_index + 1 < len(candidates)
            else len(text)
        )
        scene_index = bisect_right(scene_boundaries, candidate.start)
        next_scene = (
            scene_boundaries[scene_index]
            if scene_index < len(scene_boundaries)
            else len(text)
        )
        candidate_end = min(next_candidate, next_scene)
        candidate_end = _dialogue_action_boundary(
            text, candidate.label_end, candidate_end, candidate.label
        )
        body = text[candidate.label_end:candidate_end].strip()
        if _ACTION_DIRECTION_BODY.match(body):
            continue
        scene_id = _scene_for_offset(scenes, candidate.start)
        in_song = _offset_in_ranges(candidate.start, song_ranges)
        is_known_speaker = _is_known_speaker_label(candidate.label, known_speakers)
        if in_song and not is_known_speaker and active_speakers.get(scene_id):
            continue
        if in_song and is_known_speaker:
            candidate = _Candidate(
                candidate.start, candidate.label_end, candidate.label, False
            )
        dialogue_candidates.append(candidate)
        if is_known_speaker:
            active_speakers[scene_id] = candidate.label

    warnings: List[ParseWarning] = []
    turns: List[ParsedDialogueTurn] = []
    for candidate_index, candidate in enumerate(dialogue_candidates):
        next_candidate = (
            dialogue_candidates[candidate_index + 1].start
            if candidate_index + 1 < len(dialogue_candidates)
            else len(text)
        )
        scene_index = bisect_right(scene_boundaries, candidate.start)
        next_scene = (
            scene_boundaries[scene_index]
            if scene_index < len(scene_boundaries)
            else len(text)
        )
        end = min(next_candidate, next_scene)
        end = _dialogue_action_boundary(
            text, candidate.label_end, end, candidate.label
        )
        body = text[candidate.label_end:end].strip()
        if candidate.ambiguous or not body:
            if (
                not candidate.ambiguous
                and not body
                and _is_known_speaker_label(candidate.label, known_speakers)
            ):
                continue
            warnings.append(
                _make_warning(
                    document_id,
                    parser_version,
                    _scene_for_offset(scenes, candidate.start),
                    candidate.start,
                    candidate.label_end,
                    "ambiguous_speaker" if candidate.ambiguous else "unmatched_dialogue",
                    (
                        "Uppercase text could be a speaker cue or action; left unattributed."
                        if candidate.ambiguous
                        else "Possible speaker cue has no dialogue before the next header."
                    ),
                )
            )
            continue
        while end > candidate.start and text[end - 1].isspace():
            end -= 1
        if end <= candidate.start:
            continue
        scene_id = _scene_for_offset(scenes, candidate.start)
        turn_ordinal = len(turns)
        turn_text = text[candidate.start:end]
        turns.append(
            ParsedDialogueTurn(
                turn_id=_stable_id(
                    "turn",
                    [
                        document_id,
                        parser_version,
                        turn_ordinal,
                        scene_id,
                        candidate.start,
                        end,
                        candidate.label,
                    ],
                ),
                scene_id=scene_id,
                turn_ordinal=turn_ordinal,
                start_offset=candidate.start,
                end_offset=end,
                turn_text=turn_text,
                original_speaker_label=candidate.label,
                attribution_status="cue_parsed",
            )
        )
    return tuple(turns), tuple(warnings)


def _chunk_id(
    document_id: str,
    chunker_version: str,
    chunk_type: str,
    ordinal: int,
    spans: Sequence[SourceSpan],
) -> str:
    return _stable_id(
        "chunk",
        [
            document_id,
            chunker_version,
            chunk_type,
            ordinal,
            [[span.start_offset, span.end_offset] for span in spans],
        ],
    )


def _make_chunk(
    document_id: str,
    chunk_type: str,
    spans: Sequence[SourceSpan],
    text: str,
    ordinal: int,
    chunker_version: str,
) -> ParsedChunk:
    ordered_spans = tuple(spans)
    chunk_text = COMPOSITE_SEPARATOR.join(
        text[span.start_offset:span.end_offset] for span in ordered_spans
    )
    return ParsedChunk(
        chunk_id=_chunk_id(
            document_id, chunker_version, chunk_type, ordinal, ordered_spans
        ),
        chunker_version=chunker_version,
        chunk_ordinal=ordinal,
        chunk_type=chunk_type,
        spans=ordered_spans,
        chunk_text=chunk_text,
    )


def _spans_for_turns(turns: Sequence[ParsedDialogueTurn]) -> Tuple[SourceSpan, ...]:
    spans: List[SourceSpan] = []
    for turn in turns:
        span = SourceSpan(turn.start_offset, turn.end_offset)
        if spans and spans[-1].end_offset == span.start_offset:
            spans[-1] = SourceSpan(spans[-1].start_offset, span.end_offset)
        else:
            spans.append(span)
    return tuple(spans)


def _make_chunks(
    document_id: str,
    source_class: str,
    text: str,
    lines: Sequence[_Line],
    paragraphs: Sequence[Tuple[int, int]],
    scenes: Sequence[ParsedScene],
    turns: Sequence[ParsedDialogueTurn],
    chunker_version: str,
) -> Tuple[ParsedChunk, ...]:
    items: List[Tuple[int, str, Tuple[SourceSpan, ...]]] = []
    for scene in scenes:
        items.append(
            (
                scene.start_offset,
                "scene",
                (SourceSpan(scene.start_offset, scene.end_offset),),
            )
        )

    turns_by_scene: Dict[Optional[str], List[ParsedDialogueTurn]] = {}
    for turn in turns:
        turns_by_scene.setdefault(turn.scene_id, []).append(turn)
    for scene_turns in turns_by_scene.values():
        pending: List[ParsedDialogueTurn] = []
        pending_size = 0
        for turn in scene_turns:
            if pending and pending_size + len(turn.turn_text) > _MAX_DIALOGUE_CHUNK_CHARS:
                spans = _spans_for_turns(pending)
                items.append((spans[0].start_offset, "dialogue_exchange", spans))
                pending = []
                pending_size = 0
            pending.append(turn)
            pending_size += len(turn.turn_text)
        if pending:
            spans = _spans_for_turns(pending)
            items.append((spans[0].start_offset, "dialogue_exchange", spans))

    if source_class == "fanfiction":
        for first_index, last_index in paragraphs:
            start = lines[first_index].start
            end = lines[last_index].end
            if end > start:
                items.append((start, "prose_passage", (SourceSpan(start, end),)))

    items.sort(
        key=lambda item: (
            item[0],
            {"scene": 0, "dialogue_exchange": 1, "prose_passage": 2}[item[1]],
            item[2][-1].end_offset,
        )
    )
    chunks = []
    for ordinal, (_, chunk_type, spans) in enumerate(items):
        chunks.append(
            _make_chunk(
                document_id,
                chunk_type,
                spans,
                text,
                ordinal,
                chunker_version,
            )
        )
    return tuple(chunks)


def parse_document(
    document_id: str,
    source_class: str,
    normalized_text: str,
    parser_version: str = PARSER_VERSION,
    chunker_version: str = CHUNKER_VERSION,
    layout_spans: Sequence[Dict[str, object]] = (),
) -> ParseResult:
    """Parse one stored normalized document without guessing uncertain speakers."""
    if source_class not in ("canonical", "fanfiction"):
        raise ValueError("Unknown source class: {!r}".format(source_class))
    if "\r" in normalized_text:
        raise ValueError("Parser input must use normalized LF line endings.")
    lines = _split_lines(normalized_text)
    paragraphs = _paragraphs(lines)
    scenes = _make_scenes(document_id, normalized_text, lines, parser_version)
    turns, warnings = _parse_dialogue(
        document_id,
        source_class,
        normalized_text,
        lines,
        scenes,
        parser_version,
        layout_spans,
    )
    heading_warnings = tuple(
        _make_warning(
            document_id,
            parser_version,
            None,
            line.start,
            line.end,
            "uncertain_scene_heading",
            "Scene-like INT/EXT heading does not match a recognized format.",
        )
        for line in lines
        if (
            _UNCERTAIN_SCREENPLAY_HEADING.match(line.text.strip())
            or _UNCERTAIN_SLUG.match(line.text.strip())
        )
        and not _is_camera_direction(line.text.strip())
        and not _is_scene_heading(line.text)
        and line.end > line.start
    )
    chunks = _make_chunks(
        document_id,
        source_class,
        normalized_text,
        lines,
        paragraphs,
        scenes,
        turns,
        chunker_version,
    )
    result = ParseResult(
        document_id=document_id,
        parser_version=parser_version,
        scenes=scenes,
        dialogue_turns=turns,
        chunks=chunks,
        warnings=warnings + heading_warnings,
    )
    validate_parse_result(document_id, normalized_text, result)
    return result


def validate_parse_result(
    document_id: str, normalized_text: str, result: ParseResult
) -> None:
    """Reject invalid offsets, ordering, ownership, or composed chunk text."""
    if result.document_id != document_id:
        raise ValueError("Parse result belongs to a different document.")
    scene_ids = {scene.scene_id for scene in result.scenes}
    for scene in result.scenes:
        _validate_span(normalized_text, scene.start_offset, scene.end_offset)
        if scene.scene_text != normalized_text[scene.start_offset:scene.end_offset]:
            raise ValueError("Scene text does not match its source span.")
    for turn in result.dialogue_turns:
        _validate_span(normalized_text, turn.start_offset, turn.end_offset)
        if turn.turn_text != normalized_text[turn.start_offset:turn.end_offset]:
            raise ValueError("Dialogue turn text does not match its source span.")
        if turn.scene_id is not None and turn.scene_id not in scene_ids:
            raise ValueError("Dialogue turn references a scene from another document.")
        if turn.scene_id is not None:
            scene = next(scene for scene in result.scenes if scene.scene_id == turn.scene_id)
            if not (
                scene.start_offset <= turn.start_offset < turn.end_offset <= scene.end_offset
            ):
                raise ValueError("Dialogue turn lies outside its owning scene.")
    for chunk in result.chunks:
        if not chunk.spans:
            raise ValueError("Every chunk must have at least one source span.")
        previous_end = -1
        source_parts = []
        for span in chunk.spans:
            _validate_span(normalized_text, span.start_offset, span.end_offset)
            if span.start_offset < previous_end:
                raise ValueError("Chunk spans must be ordered and non-overlapping.")
            previous_end = span.end_offset
            source_parts.append(normalized_text[span.start_offset:span.end_offset])
        expected = COMPOSITE_SEPARATOR.join(source_parts)
        if chunk.chunk_text != expected:
            raise ValueError("Chunk text does not match its ordered source spans.")
    for warning in result.warnings:
        _validate_span(normalized_text, warning.start_offset, warning.end_offset)
        if warning.scene_id is not None and warning.scene_id not in scene_ids:
            raise ValueError("Parse warning references a scene from another document.")


def _validate_span(text: str, start_offset: int, end_offset: int) -> None:
    if not (0 <= start_offset < end_offset <= len(text)):
        raise ValueError(
            "Invalid source span [{}, {}) for document length {}.".format(
                start_offset, end_offset, len(text)
            )
        )
