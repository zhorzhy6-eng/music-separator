"""
Транскрипция минуса в фортепианную партию через MuScriptor.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pretty_midi

logger = logging.getLogger(__name__)

PIANO_PROGRAM = 0  # Acoustic Grand Piano


class PianoTranscriberError(Exception):
    """Ошибка транскрипции в пианино."""


class PianoTranscriber:
    """
    Обёртка над MuScriptor 0.3.x + сведение в одну фортепианную дорожку.
    """

    def __init__(self, model_size: str = "small") -> None:
        self.model_size = model_size
        self._model = None
        logger.info("PianoTranscriber init", extra={"ctx_model": model_size})

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from muscriptor import TranscriptionModel
            logger.info("Загружаю MuScriptor: %s", self.model_size)
            self._model = TranscriptionModel.load_model(self.model_size)
            logger.info("MuScriptor загружен")
        except Exception as e:
            logger.exception("Не удалось загрузить MuScriptor")
            raise PianoTranscriberError(f"MuScriptor не загрузился: {e}") from e

    def transcribe(
        self,
        input_audio: str | Path,
        output_midi: str | Path,
        progress_cb=None,
    ) -> Path:
        self._ensure_model()
        in_path = Path(input_audio)
        out_path = Path(output_midi)

        if not in_path.exists():
            raise PianoTranscriberError(f"Аудио не найдено: {in_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            progress_cb(0.05, "Транскрипция нейросетью (MuScriptor)")

        # --- Шаг 1: MuScriptor → многодорожечный MIDI ---
        tmp_multi = out_path.with_name(out_path.stem + "_multi.mid")
        try:
            # 0.3.0: возвращает tuple (bytes, BeatGrid|None)
            result = self._model.transcribe_and_postprocess(str(in_path))

            if isinstance(result, tuple):
                data = result[0]
            else:
                data = result

            if not isinstance(data, (bytes, bytearray)):
                raise PianoTranscriberError(
                    f"MuScriptor вернул не bytes, а {type(data).__name__}"
                )

            tmp_multi.write_bytes(data)
            logger.info("MuScriptor готов", extra={"ctx_multi": str(tmp_multi)})

        except PianoTranscriberError:
            raise
        except Exception as e:
            logger.exception("Ошибка MuScriptor")
            raise PianoTranscriberError(f"MuScriptor упал: {e}") from e

        if progress_cb:
            progress_cb(0.85, "Сведение в фортепиано")

        # --- Шаг 2: свести всё в одну фортепианную дорожку ---
        try:
            src = pretty_midi.PrettyMIDI(str(tmp_multi))
            dst = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=PIANO_PROGRAM)

            total = 0
            for instr in src.instruments:
                if instr.is_drum:
                    continue
                for note in instr.notes:
                    piano.notes.append(pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=note.end,
                    ))
                    total += 1

            # Поднимаем низкие ноты в комфортный регистр пианино
            for note in piano.notes:
                if note.pitch < 48:      # ниже C3 — бас/саб-бас
                    note.pitch += 24     # +2 октавы
                elif note.pitch < 60:    # C3–B3 — низкая середина
                    note.pitch += 12     # +1 октава

            # Если после сдвига pitch > 127 — обрезаем
            for note in piano.notes:
                if note.pitch > 127:
                    note.pitch = 127

            dst.instruments.append(piano)
            dst.write(str(out_path))
            logger.info("Фортепианная партия готова",
                        extra={"ctx_out": str(out_path), "ctx_notes": total})
        except Exception as e:
            logger.exception("Ошибка сведения в фортепиано")
            raise PianoTranscriberError(f"Сведение упало: {e}") from e

        if progress_cb:
            progress_cb(1.0, "Готово")

        return out_path