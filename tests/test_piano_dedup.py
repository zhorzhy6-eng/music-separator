"""
Проверки дедупликации нот фортепианной партии.

Было в QA: 748 нот-дублей после сведения дорожек MuScriptor в одну партию.
Дубль в MIDI звучит громче и с фазированием, поэтому их надо снимать до записи,
а перекрывающиеся ноты одной высоты — склеивать (одна клавиша не звучит дважды).
"""
from __future__ import annotations

import pretty_midi

from core.piano_transcriber import PIANO_MIN_PITCH, merge_piano_notes


def test_duplicates_are_removed():
    """
    748 полных дублей → 0.

    Ноты идут с шагом 0.6 с при длине 0.5 с: соседние не перекрываются, поэтому
    после снятия дублей должно остаться ровно 748 отдельных нот.
    """
    raw = []
    for i in range(748):
        start = i * 0.6
        end = start + 0.5
        raw.append((60, start, end, 90))
        raw.append((60, start, end, 90))     # ровно такой же дубль

    merged, stats = merge_piano_notes(raw)

    assert stats["total"] == 748 * 2
    assert stats["removed_duplicates"] == 748
    assert stats["merged_overlaps"] == 0
    assert len(merged) == 748
    keys = [(p, s, e) for p, s, e, _ in merged]
    assert len(keys) == len(set(keys))


def test_overlapping_same_pitch_are_merged():
    """Две одновременные ноты одной высоты становятся одной длинной."""
    raw = [
        (64, 0.0, 0.5, 80),
        (64, 0.2, 0.8, 100),
    ]
    merged, stats = merge_piano_notes(raw)

    assert len(merged) == 1
    pitch, start, end, vel = merged[0]
    assert pitch == 64
    assert start == 0.0
    assert end == 0.8
    assert vel == 100                      # берём громкость сильнейшей
    assert stats["merged_overlaps"] == 1


def test_low_notes_are_shifted_into_range():
    """Ноты ниже C3 поднимаются на октавы вверх, выше 127 не уходят."""
    raw = [
        (24, 0.0, 0.5, 80),      # C1 → должен стать ≥ C3
        (0, 1.0, 1.5, 80),       # совсем низкая
        (120, 2.0, 2.5, 80),     # высокая — не должна вылезти за предел
    ]
    merged, _ = merge_piano_notes(raw)

    for pitch, _, _, _ in merged:
        assert PIANO_MIN_PITCH <= pitch <= 127


def test_notes_are_sorted_by_start():
    raw = [
        (60, 2.0, 2.5, 80),
        (62, 0.0, 0.5, 80),
        (64, 1.0, 1.5, 80),
    ]
    merged, _ = merge_piano_notes(raw)
    starts = [s for _, s, _, _ in merged]
    assert starts == sorted(starts)


def test_independent_notes_survive():
    """Разные высоты и разные моменты не должны схлопываться."""
    raw = [
        (60, 0.0, 0.5, 80),
        (64, 0.0, 0.5, 80),
        (67, 0.5, 1.0, 80),
    ]
    merged, stats = merge_piano_notes(raw)
    assert len(merged) == 3
    assert stats["removed_duplicates"] == 0
    assert stats["merged_overlaps"] == 0
    assert stats["unique_pitches"] == 3


def test_result_survives_midi_roundtrip(tmp_path):
    """После записи и чтения MIDI дублей не появляется."""
    raw = [(60, 0.0, 0.5, 90), (60, 0.0, 0.5, 90), (64, 0.1, 0.6, 90)]
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    merged, _ = merge_piano_notes(raw, midi=midi)
    for pitch, start, end, vel in merged:
        piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                            start=start, end=end))
    midi.instruments.append(piano)

    path = tmp_path / "piano.mid"
    midi.write(str(path))
    back = pretty_midi.PrettyMIDI(str(path))
    notes = back.instruments[0].notes

    keys = [(n.pitch, round(n.start, 4), round(n.end, 4)) for n in notes]
    assert len(keys) == len(set(keys)), "дубли вернулись после записи файла"
