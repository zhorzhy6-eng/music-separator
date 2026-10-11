"""
Сведение многодорожечного MIDI в одну фортепианную партию.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pretty_midi

logger = logging.getLogger(__name__)

PIANO_PROGRAM = 0  # Acoustic Grand Piano


class PianoReducer:
    def reduce(self, input_midi: str | Path, output_midi: str | Path) -> Path:
        in_path = Path(input_midi)
        out_path = Path(output_midi)
        if not in_path.exists():
            raise FileNotFoundError(f"MIDI не найден: {in_path}")

        src = pretty_midi.PrettyMIDI(str(in_path))
        dst = pretty_midi.PrettyMIDI()

        # Одна дорожка — фортепиано
        piano = pretty_midi.Instrument(program=PIANO_PROGRAM)

        total_notes = 0
        for instr in src.instruments:
            if instr.is_drum:
                continue  # барабаны пропускаем
            for note in instr.notes:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
                piano.notes.append(new_note)
                total_notes += 1

        # Поднять слишком низкие ноты на октаву вверх (бас)
        for note in piano.notes:
            if note.pitch < 48:  # ниже C3
                note.pitch += 12

        dst.instruments.append(piano)
        dst.write(str(out_path))

        logger.info("Фортепианная редукция завершена",
                    extra={"ctx_in": str(in_path), "ctx_out": str(out_path),
                           "ctx_notes": total_notes})
        return out_path