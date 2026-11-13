from dataclasses import replace
import sqlite3

import pytest

from scriptlab.config import ProjectConfig
from scriptlab.database import (
    DATABASE_FILENAME,
    import_sources,
    inspect_document,
    parse_documents,
)
from scriptlab.parser import (
    COMPOSITE_SEPARATOR,
    SourceSpan,
    parse_document,
    validate_parse_result,
)


SCRIPT_TEXT = (
    "INT. ROOM - DAY\n\n"
    "BUFFY\n"
    "We should go.\n\n"
    "A VERY STRANGE ACTION HAPPENS\n"
    "Buffy grabs her bag.\n\n"
    "WILLOW (O.S.)\n"
    "Wait for me.\n"
)


def make_config(base):
    canonical = base / "scripts"
    fanfiction = base / "fanfiction"
    canonical.mkdir(parents=True, exist_ok=True)
    fanfiction.mkdir(parents=True, exist_ok=True)
    return ProjectConfig(
        project_name="test",
        canonical_directories=(canonical,),
        fanfiction_directory=fanfiction,
        storage_directory=base / "data",
        embedding_model="test-model",
        ollama_url="http://127.0.0.1:11434",
    )


def load_document_id(config, filename):
    database_path = config.storage_directory / DATABASE_FILENAME
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            "SELECT document_id FROM documents WHERE filename = ?", (filename,)
        ).fetchone()[0]


def test_script_scenes_dialogue_and_composite_citations_are_exact():
    result = parse_document("doc-script", "canonical", SCRIPT_TEXT)

    assert len(result.scenes) == 1
    scene = result.scenes[0]
    assert scene.scene_heading == "INT. ROOM - DAY"
    assert SCRIPT_TEXT[scene.start_offset : scene.end_offset] == scene.scene_text
    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "BUFFY",
        "WILLOW (O.S.)",
    ]
    assert all(turn.speaker_character_id is None for turn in result.dialogue_turns)
    assert all(turn.attribution_status == "cue_parsed" for turn in result.dialogue_turns)
    for turn in result.dialogue_turns:
        assert SCRIPT_TEXT[turn.start_offset : turn.end_offset] == turn.turn_text
        assert turn.scene_id == scene.scene_id

    exchange = next(
        chunk for chunk in result.chunks if chunk.chunk_type == "dialogue_exchange"
    )
    assert len(exchange.spans) == 2
    for span in exchange.spans:
        assert SCRIPT_TEXT[span.start_offset : span.end_offset]
    assert "Buffy grabs her bag." not in exchange.chunk_text
    assert COMPOSITE_SEPARATOR in exchange.chunk_text


def test_ambiguous_capitalized_action_warns_and_fanfiction_stays_unattributed():
    ambiguous = "INT. ROOM - DAY\n\nTHE DOOR IS OPEN\nShe steps back.\n"
    result = parse_document("doc-action", "canonical", ambiguous)

    assert not result.dialogue_turns
    assert result.warnings[0].warning_code == "ambiguous_speaker"
    assert result.warnings[0].start_offset == ambiguous.index("THE DOOR")
    assert ambiguous[
        result.warnings[0].start_offset : result.warnings[0].end_offset
    ] == "THE DOOR IS OPEN"

    fanfiction = 'Chapter One\n\nBUFFY\n"I miss you," Willow said.\n'
    prose_result = parse_document("doc-fic", "fanfiction", fanfiction)

    assert not prose_result.dialogue_turns
    assert not prose_result.warnings
    assert {chunk.chunk_type for chunk in prose_result.chunks} == {
        "scene",
        "prose_passage",
    }


def test_parser_ids_are_repeatable_and_change_with_chunker_version():
    first = parse_document("doc-script", "canonical", SCRIPT_TEXT)
    repeated = parse_document("doc-script", "canonical", SCRIPT_TEXT)
    revised = parse_document(
        "doc-script", "canonical", SCRIPT_TEXT, chunker_version="scriptlab-chunker-v17"
    )

    assert first == repeated
    assert [chunk.chunk_id for chunk in first.chunks] != [
        chunk.chunk_id for chunk in revised.chunks
    ]
    assert [chunk.chunk_ordinal for chunk in first.chunks] == list(
        range(len(first.chunks))
    )


def test_parse_validation_rejects_missing_invalid_and_misordered_spans():
    result = parse_document("doc-script", "canonical", SCRIPT_TEXT)
    chunk_index = next(
        index
        for index, chunk in enumerate(result.chunks)
        if chunk.chunk_type == "dialogue_exchange"
    )
    chunk = result.chunks[chunk_index]
    mutations = (
        replace(chunk, spans=()),
        replace(chunk, spans=(SourceSpan(len(SCRIPT_TEXT), len(SCRIPT_TEXT) + 1),)),
        replace(chunk, spans=(chunk.spans[0], chunk.spans[0])),
        replace(chunk, spans=tuple(reversed(chunk.spans))),
    )

    for malformed in mutations:
        chunks = list(result.chunks)
        chunks[chunk_index] = malformed
        with pytest.raises(ValueError):
            validate_parse_result(
                result.document_id, SCRIPT_TEXT, replace(result, chunks=tuple(chunks))
            )

    with pytest.raises(ValueError, match="different document"):
        validate_parse_result("another-document", SCRIPT_TEXT, result)


def test_parse_persists_provenance_warnings_and_inspect_snippets(tmp_path):
    config = make_config(tmp_path)
    script = config.canonical_directories[0] / "episode.txt"
    script.write_text(
        "INT. ROOM - DAY\n\nBUFFY\nWait.\n\nTHE DOOR IS OPEN\n"
        "She steps back.\n\nWILLOW (O.S.)\nComing.\n",
        encoding="utf-8",
    )
    fic = config.fanfiction_directory / "story.txt"
    fic.write_text('Chapter One\n\nBUFFY\n"I miss you," Willow said.\n', encoding="utf-8")
    import_sources(config)
    script_id = load_document_id(config, script.name)
    fic_id = load_document_id(config, fic.name)

    summary = parse_documents(config, script_id)
    inspection = inspect_document(config, script_id)
    fic_summary = parse_documents(config, fic_id)
    fic_inspection = inspect_document(config, fic_id)

    assert summary.documents_parsed == 1
    assert summary.dialogue_turns == 2
    assert summary.warnings == 1
    assert inspection["dialogue_turns"][0]["original_speaker_label"] == "BUFFY"
    assert inspection["dialogue_turns"][0]["speaker_character_id"] is None
    assert inspection["warnings"][0]["warning_code"] == "ambiguous_speaker"
    assert inspection["warnings"][0]["source_snippet"] == "THE DOOR IS OPEN"
    exchange = next(
        chunk
        for chunk in inspection["chunks"]
        if chunk["chunk_type"] == "dialogue_exchange"
    )
    assert len(exchange["spans"]) == 2
    assert all(span["source_snippet"] for span in exchange["spans"])
    assert "She steps back." not in exchange["chunk_text"]
    assert fic_summary.dialogue_turns == 0
    assert not fic_inspection["dialogue_turns"]
    assert all(
        chunk["chunk_type"] in ("scene", "prose_passage")
        for chunk in fic_inspection["chunks"]
    )


def test_nonstandard_scene_slugs_are_scene_boundaries_not_warnings():
    text = (
        "INT ROOM - NIGHT\n\n"
        "MARGO - ON THE PHONE IN HER HIGH RISE - DAY\n\n"
        "EXT - ROOF\n\n"
        "EXT: The Bronze. Night.\n\n"
        "A REALLY LONG LENS SHOT - BINOC MATT - DAY\n\n"
        "BUFFY\nStay.\n"
    )

    result = parse_document("doc-heading", "canonical", text)

    assert not result.warnings
    assert [
        scene.scene_heading for scene in result.scenes
    ] == [
        "INT ROOM - NIGHT",
        "MARGO - ON THE PHONE IN HER HIGH RISE - DAY",
        "EXT - ROOF",
        "EXT: The Bronze. Night.",
    ]
    assert all(
        "A REALLY LONG LENS SHOT" not in scene.scene_heading
        for scene in result.scenes
        if scene.scene_heading is not None
    )


def test_transition_and_camera_labels_are_not_speaker_cues():
    text = (
        "INT. ROOM - DAY\n\nBLACKOUT.\n\n"
        "ANGLE - SHOOTING OVER THE TABLE\n\nTHE END\n"
    )

    result = parse_document("doc-transitions", "canonical", text)

    assert not result.dialogue_turns
    assert not result.warnings


@pytest.mark.parametrize(
    "marker",
    ("OMITTED", "END OF SHOW", "END OF EPISODE", "RESUME", "FADE TO BLACK.", "BLACK"),
)
def test_production_end_markers_do_not_generate_speaker_warnings(marker):
    text = "INT. ROOM - DAY\n\n{}\n".format(marker)

    result = parse_document("doc-end-marker", "canonical", text)

    assert not result.dialogue_turns
    assert not result.warnings


@pytest.mark.parametrize(
    "direction",
    (
        "WE SEE",
        "PULL BACK",
        "CLOSE SHOTS",
        "WIDER",
        "CLOSES ON THE COUPLE",
        "VIEW FROM BUFFY'S WINDOW",
        "WIDEN TO SHOW",
        "END OF PART ONE",
        "30 THRU 32 OMITTED",
        "RESUME BUFFY AND FRIENDS",
        "BUFFY'S P.O.V.",
        "FROM BUFFY'S POINT OF VIEW",
        "DRAMATIC ANGLE - BUFFY'S SAVIOR",
        "HIS THUMB",
        "A HARVEST MOON",
        "THE WOMAN'S DEAD FACE",
        "A DARK FIGURE",
        "THE KILLER",
        "ANGEL'S CAR",
        "CLOSE - KATE'S HAND",
    ),
)
def test_camera_direction_triggers_are_not_attributed_as_speakers(direction):
    text = "INT. ROOM - DAY\n\n{}\nBuffy enters.\n\nBUFFY\nHi.\n".format(
        direction
    )

    result = parse_document("doc-camera", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == ["BUFFY"]
    assert not result.warnings


def test_character_action_sentences_and_group_reactions_are_not_dialogue():
    text = (
        "INT. GYM - DAY\n\n"
        "CORDELIA IS STANDING RIGHT NEXT TO HER\n\n"
        "BUFFY FLIES THROUGH THE AIR, TACKLES CORDELIA\n\n"
        "WILLOW AND XANDER WINCE\n\n"
        "WILLOW AND XANDER\n\nReact to Buffy.\n\n"
        "WILLOW\nWe better get her out of there.\n\n"
        "WILLOW AND XANDER\n\n"
        "XANDER\nYeah.\n"
    )

    result = parse_document("doc-character-actions", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "WILLOW",
        "XANDER",
    ]
    assert not result.warnings


def test_common_role_and_delivery_annotations_remain_original_cues():
    text = (
        "INT. ROOM - DAY\n\n"
        "JOE'S VOICE ON MACHINE\nLeave a message.\n\n"
        "GUARD (IN SPANISH, SUB-TITLED)\nStop.\n\n"
        "SMART YOUNG LAWYER\nI object.\n"
        "M. C.'S VOICE\nLadies and gentlemen.\n\n"
        "M.C.'S VOICE\nFrom Sunnydale.\n"
    )

    result = parse_document("doc-cues", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "JOE'S VOICE ON MACHINE",
        "GUARD (IN SPANISH, SUB-TITLED)",
        "SMART YOUNG LAWYER",
        "M. C.'S VOICE",
        "M.C.'S VOICE",
    ]
    assert not result.warnings


def test_speaker_cue_separated_from_dialogue_parses_across_blank_lines():
    text = "INT. ROOM - DAY\n\nBUFFY\n\nWait here.\n"

    result = parse_document("doc-separated", "canonical", text)

    assert len(result.dialogue_turns) == 1
    assert result.dialogue_turns[0].original_speaker_label == "BUFFY"
    assert result.dialogue_turns[0].turn_text == "BUFFY\n\nWait here."


def test_speaker_block_stops_at_camera_direction_between_blank_lines():
    text = (
        "INT. ROOM - DAY\n\n"
        "WESLEY\n\n(thrown)\nGot?\n\nCLOSE SHOTS\n\n"
        "CORDELIA\nWhat do you want?\n"
    )

    result = parse_document("doc-header-block", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "WESLEY",
        "CORDELIA",
    ]
    assert result.dialogue_turns[0].turn_text == (
        "WESLEY\n\n(thrown)\nGot?"
    )
    assert result.dialogue_turns[0].end_offset <= result.dialogue_turns[1].start_offset


def test_speaker_block_stops_at_next_scene_heading():
    text = "INT. ROOM - DAY\n\nBUFFY\nStay here.\n\nEXT. STREET - NIGHT\n\nXANDER\nOkay.\n"

    result = parse_document("doc-scene-boundary", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "BUFFY",
        "XANDER",
    ]
    assert "EXT. STREET - NIGHT" not in result.dialogue_turns[0].turn_text
    assert result.dialogue_turns[0].scene_id != result.dialogue_turns[1].scene_id


def test_dialogue_turns_stop_before_action_paragraphs():
    text = (
        "INT. ALLEY - NIGHT\n\n"
        "KATE\nWhere?\n\n"
        "The uniform leads her to the body.\n\n"
        "KATE\nIt's the same guy.\n\n"
        "The others follow her gaze.\n"
    )

    result = parse_document("doc-dialogue-action-boundary", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "KATE",
        "KATE",
    ]
    assert [turn.turn_text for turn in result.dialogue_turns] == [
        "KATE\nWhere?",
        "KATE\nIt's the same guy.",
    ]


def test_group_speaker_cue_followed_by_action_is_not_dialogue():
    text = (
        "INT. ROOM - DAY\n\n"
        "BUFFY AND WILLOW\n\nFlank him on the couch.\n\n"
        "XANDER\nWhat are you doing?\n"
    )

    result = parse_document("doc-group-action", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "XANDER"
    ]


def test_named_character_action_paragraph_stops_previous_dialogue_turn():
    text = (
        "INT. STREET - NIGHT\n\n"
        "WILLOW\nCome and get it.\n\n"
        "Andrew rises and Willow takes a step back.\n\n"
        "XANDER\nI got him!\n"
    )

    result = parse_document("doc-named-action", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "WILLOW",
        "XANDER",
    ]
    assert [turn.turn_text for turn in result.dialogue_turns] == [
        "WILLOW\nCome and get it.",
        "XANDER\nI got him!",
    ]


def test_object_insert_followed_by_action_is_not_a_speaker():
    text = (
        "EXT. ALLEY - NIGHT\n\n"
        "GYM BAG\n\n"
        "Falls from a height, hits the pavement.\n\n"
        "SPIVEY\nWhat was that?\n"
    )

    result = parse_document("doc-object-insert", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "SPIVEY"
    ]
    assert any("GYM BAG" in scene.scene_text for scene in result.scenes)
    assert any("Falls from a height" in chunk.chunk_text for chunk in result.chunks)


def test_left_aligned_cue_stops_previous_centered_dialogue():
    text = (
        "INT. ROOM - DAY\n\n"
        "XANDER\nIs she dying?\n\n"
        "BUFFY AND WILLOW\n\nFlank him on the couch.\n\n"
        "BUFFY\nShe is singing.\n"
    )
    first_speaker = text.index("XANDER")
    first_end = text.index("BUFFY AND WILLOW")
    last_speaker = text.rindex("BUFFY")
    layout_spans = (
        {"start_offset": first_speaker, "end_offset": first_end, "role": "center"},
        {"start_offset": last_speaker, "end_offset": len(text), "role": "center"},
    )

    result = parse_document(
        "doc-left-cue-boundary", "canonical", text, layout_spans=layout_spans
    )

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "XANDER",
        "BUFFY",
    ]
    assert [turn.turn_text for turn in result.dialogue_turns] == [
        "XANDER\nIs she dying?",
        "BUFFY\nShe is singing.",
    ]


def test_standalone_parenthetical_and_interrupted_fragment_are_not_speaker_cues():
    text = (
        "INT. LIBRARY - DAY\n\n"
        "ANGEL\n(O.S.)\nIt can't be.\n\n"
        "XANDER\n(YELLING)\nHEY!\n\n"
        "GILES\nC --\n\n"
        "BUFFY\nOkay.\n"
    )

    result = parse_document("doc-paren", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "ANGEL",
        "XANDER",
        "GILES",
        "BUFFY",
    ]
    assert not result.warnings


@pytest.mark.parametrize(
    "label",
    [
        "LOS ANGELES SKYLINE - NIGHT",
        "LAIR - SHORT TIME LATER",
        "ANGEL'S POV - NIGHT SCOPE MATTE",
        "ANGEL'S P.O.V.",
        "ON XANDER AND WILLOW",
        "CLOSE ON ANGEL",
        "BACK ON THE CROWD",
        "END MONTAGE",
        "END OF SEASON",
    ],
)
def test_time_of_day_slugs_pov_and_on_subject_shots_are_not_speaker_cues(label):
    text = f"INT. ROOM - DAY\n\nBUFFY\nHi.\n\n{label}\n\nGILES\nHello.\n"

    result = parse_document("doc-shot", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "BUFFY",
        "GILES",
    ]
    assert not result.warnings


def test_wrapped_voiceover_cue_does_not_turn_first_person_pronoun_into_speaker():
    text = (
        "INT. ROOM - DAY\n\n"
        "CORDELIA (V.O.)\nI\nbelieve in Los Angeles...\n\n"
        "KATE (V.\nO.)\nOur suspect will be a white male.\n\n"
        "BUFFY\nStay here.\n"
    )

    result = parse_document("doc-wrapped-voiceover", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "CORDELIA (V.O.)",
        "KATE (V.O.)",
        "BUFFY",
    ]
    assert result.dialogue_turns[0].turn_text == (
        "CORDELIA (V.O.)\nI\nbelieve in Los Angeles..."
    )
    assert result.dialogue_turns[1].turn_text.startswith("KATE (V.\nO.)\n")
    assert not result.warnings


def test_sung_uppercase_lines_stay_with_known_speaker_in_song_block():
    text = (
        "INT. STAGE - NIGHT\n\n"
        "BUFFY\nI'm here.\n\n"
        "Buffy begins to sing:\n\n"
        "BUFFY\nEVERY SINGLE NIGHT\nTHE SAME ARRANGEMENT\nBLOWS\n\n"
        "The number is over.\n\n"
        "XANDER\nAre you okay?\n"
    )

    result = parse_document("doc-song-block", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "BUFFY",
        "BUFFY",
        "XANDER",
    ]
    assert "EVERY SINGLE NIGHT" in result.dialogue_turns[1].turn_text
    assert "THE SAME ARRANGEMENT" in result.dialogue_turns[1].turn_text
    assert "BLOWS" in result.dialogue_turns[1].turn_text
    assert not result.warnings


def test_song_action_cue_and_combined_known_speakers_are_recognized():
    text = (
        "INT. STAGE - NIGHT\n\n"
        "WILLOW\nI'm ready.\n\n"
        "ANYA\nLet's go.\n\n"
        "Willow and Anya take the song, explaining:\n\n"
        "WILLOW/ANYA\nWE HAVE A PLAN\n\n"
        "The number is over.\n\n"
        "BUFFY\nThat was strange.\n"
    )

    result = parse_document("doc-combined-song", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "WILLOW",
        "ANYA",
        "WILLOW/ANYA",
        "BUFFY",
    ]
    assert "WE HAVE A PLAN" in result.dialogue_turns[2].turn_text
    assert not result.warnings


def test_song_marker_after_lyrics_extends_to_previous_known_cue():
    text = (
        "INT. STAGE - NIGHT\n\n"
        "GILES\nWe should talk.\n\n"
        "WILLOW\nI agree.\n\n"
        "GILES\nI'VE GOT A THEORY\nTHAT IT'S A DEMON\nA DANCING DEMON\n\n"
        "WILLOW\nSOME KID IS DREAMIN'\nAND WE'RE ALL IN IT\n\n"
        "Buffy takes the song, explaining:\n\n"
        "BUFFY\nHERE'S WHAT WE DO\n\n"
        "The number is over.\n\n"
        "XANDER\nThat was strange.\n"
    )

    result = parse_document("doc-mid-song-marker", "canonical", text)

    assert [turn.original_speaker_label for turn in result.dialogue_turns] == [
        "GILES",
        "WILLOW",
        "GILES",
        "WILLOW",
        "BUFFY",
        "XANDER",
    ]
    assert "I'VE GOT A THEORY" in result.dialogue_turns[2].turn_text
    assert "SOME KID IS DREAMIN'" in result.dialogue_turns[3].turn_text
    assert not result.warnings


def test_dialogue_turn_trims_trailing_whitespace_at_scene_boundary():
    text = "INT. ROOM - DAY\n\nBUFFY\nStay here.   \n"

    result = parse_document("doc-trailing-space", "canonical", text)

    scene = result.scenes[0]
    turn = result.dialogue_turns[0]
    assert scene.start_offset <= turn.start_offset < turn.end_offset <= scene.end_offset
    assert turn.turn_text == "BUFFY\nStay here."


def test_parse_is_idempotent_and_chunker_version_replaces_derived_rows(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "episode.txt"
    source.write_text(SCRIPT_TEXT, encoding="utf-8")
    import_sources(config)
    document_id = load_document_id(config, source.name)
    database_path = config.storage_directory / DATABASE_FILENAME

    parse_documents(config, document_id)
    with sqlite3.connect(database_path) as connection:
        initial = connection.execute(
            "SELECT chunk_id, chunk_type, chunker_version, chunk_ordinal "
            "FROM chunks WHERE document_id = ? ORDER BY chunk_ordinal",
            (document_id,),
        ).fetchall()
        initial_spans = connection.execute(
            "SELECT chunk_id, span_ordinal, start_offset, end_offset "
            "FROM chunk_spans ORDER BY chunk_id, span_ordinal"
        ).fetchall()
        original_text = connection.execute(
            "SELECT normalized_text FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]

    parse_documents(config, document_id)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT chunk_id, chunk_type, chunker_version, chunk_ordinal "
            "FROM chunks WHERE document_id = ? ORDER BY chunk_ordinal",
            (document_id,),
        ).fetchall() == initial
        assert connection.execute(
            "SELECT chunk_id, span_ordinal, start_offset, end_offset "
            "FROM chunk_spans ORDER BY chunk_id, span_ordinal"
        ).fetchall() == initial_spans

    parse_documents(config, document_id, chunker_version="scriptlab-chunker-v17")
    with sqlite3.connect(database_path) as connection:
        revised = connection.execute(
            "SELECT chunk_id, chunker_version, chunk_ordinal FROM chunks "
            "WHERE document_id = ? ORDER BY chunk_ordinal",
            (document_id,),
        ).fetchall()
        current_text = connection.execute(
            "SELECT normalized_text FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]

    assert all(row[1] == "scriptlab-chunker-v17" for row in revised)
    assert [row[0] for row in revised] != [row[0] for row in initial]
    assert current_text == original_text


def test_failed_document_parse_preserves_previous_derived_rows(tmp_path):
    config = make_config(tmp_path)
    source = config.canonical_directories[0] / "episode.txt"
    source.write_text(SCRIPT_TEXT, encoding="utf-8")
    import_sources(config)
    document_id = load_document_id(config, source.name)
    parse_documents(config, document_id)
    database_path = config.storage_directory / DATABASE_FILENAME
    with sqlite3.connect(database_path) as connection:
        before = connection.execute(
            "SELECT chunk_id, chunk_text FROM chunks WHERE document_id = ? "
            "ORDER BY chunk_ordinal",
            (document_id,),
        ).fetchall()
        connection.execute(
            "CREATE TRIGGER reject_chunk_insert BEFORE INSERT ON chunks "
            "BEGIN SELECT RAISE(ABORT, 'forced failure'); END"
        )

    summary = parse_documents(config, document_id)

    assert summary.errors
    with sqlite3.connect(database_path) as connection:
        after = connection.execute(
            "SELECT chunk_id, chunk_text FROM chunks WHERE document_id = ? "
            "ORDER BY chunk_ordinal",
            (document_id,),
        ).fetchall()
    assert after == before
