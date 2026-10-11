"""
Транскрипция минуса в фортепианную партию через MuScriptor.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pretty_midi

logger = logging.getLogger(__name__)

PIANO_PROGRAM = 0  # Acoustic Grand Piano
PIANO_MIN_PITCH = 48  # C3 — ниже этого нота поднимается на октаву вверх


class PianoTranscriberError(Exception):
    """Ошибка транскрипции в пианино."""


def merge_piano_notes(
    raw: list[tuple[int, float, float, int]],
    midi: pretty_midi.PrettyMIDI | None = None,
) -> tuple[list[tuple[int, float, float, int]], dict]:
    """
    Сводит ноты всех дорожек в одну фортепианную партию.

    Шаги (вынесены из transcribe без изменения логики — чтобы это можно было
    проверить тестом, а не только глазами на большом файле):

      1. Низкие ноты поднимаются в комфортный регистр ровно на столько октав,
         сколько нужно (жёсткий шаг +24/+12 склеивал бас и гитару в одни высоты).
      2. Полные дубли (та же высота, то же начало и конец) убираются. Время
         сравнивается В ТИКАХ MIDI, а не в секундах: две ноты, различавшиеся на
         десятые доли миллисекунды, после записи становятся одной и той же нотой.
      3. Перекрывающиеся ноты одной высоты склеиваются: MIDI хранит пары
         note_on/note_off, и две одновременные ноты одной высоты при чтении
         распадаются на дубли. На фортепиано одна клавиша не звучит дважды.

    Возвращает (список нот, статистика).
    """
    if midi is None:
        midi = pretty_midi.PrettyMIDI()

    total = len(raw)

    # --- 1) подъём низких нот ---
    shifted: list[tuple[int, float, float, int]] = []
    for pitch, start, end, vel in raw:
        while pitch < PIANO_MIN_PITCH:
            pitch += 12
        if pitch > 127:
            pitch = 127
        shifted.append((pitch, start, end, vel))

    # --- 2) полные дубли ---
    seen: set[tuple[int, int, int]] = set()
    unique: list[tuple[int, float, float, int]] = []
    for pitch, start, end, vel in shifted:
        key = (pitch, int(midi.time_to_tick(start)), int(midi.time_to_tick(end)))
        if key in seen:
            continue
        seen.add(key)
        unique.append((pitch, start, end, vel))

    removed = total - len(unique)

    # --- 3) склейка перекрытий ---
    by_pitch: dict[int, list[tuple[int, float, float, int]]] = {}
    for note in unique:
        by_pitch.setdefault(note[0], []).append(note)

    merged: list[tuple[int, float, float, int]] = []
    overlapped = 0
    for pitch in sorted(by_pitch):
        group = sorted(by_pitch[pitch], key=lambda x: x[1])
        cur_p, cur_s, cur_e, cur_v = group[0]
        for p2, s2, e2, v2 in group[1:]:
            if s2 < cur_e:                      # ноты звучат одновременно
                cur_e = max(cur_e, e2)
                cur_v = max(cur_v, v2)
                overlapped += 1
            else:
                merged.append((cur_p, cur_s, cur_e, cur_v))
                cur_p, cur_s, cur_e, cur_v = p2, s2, e2, v2
        merged.append((cur_p, cur_s, cur_e, cur_v))
    merged.sort(key=lambda x: x[1])

    stats = {
        "total": total,
        "unique": len(unique),
        "removed_duplicates": removed,
        "merged_overlaps": overlapped,
        "result": len(merged),
        "unique_pitches": len({p for p, _, _, _ in merged}),
    }
    return merged, stats


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
            # MuScriptor кладёт темповые события не на нулевую дорожку — формально
            # это невалидный MIDI, и pretty_midi про это предупреждает. Файл при
            # этом читается нормально, поэтому предупреждение гасим и говорим об
            # этом один раз в лог.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                src = pretty_midi.PrettyMIDI(str(tmp_multi))

            quirk = False
            for w in caught:
                if "Tempo, Key or Time signature" in str(w.message):
                    quirk = True
                else:
                    # чужие предупреждения не прячем
                    warnings.warn(w.message, w.category, stacklevel=2)
            if quirk:
                logger.info(
                    "_piano_multi.mid: темповые события на ненулевой дорожке "
                    "(особенность MuScriptor) — это не ошибка, файл прочитан")

            dst = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=PIANO_PROGRAM)

            # 1) Собираем ноты всех небарабанных дорожек
            raw: list[tuple[int, float, float, int]] = []
            for instr in src.instruments:
                if instr.is_drum:
                    continue
                for note in instr.notes:
                    raw.append((int(note.pitch), float(note.start),
                                float(note.end), int(note.velocity)))

            # 2-4) Сводим в одну партию: подъём низких нот, снятие дублей,
            #      склейка перекрытий. Вся эта логика — в merge_piano_notes,
            #      чтобы её можно было проверить тестом (tests/test_piano_dedup.py).
            merged, stats = merge_piano_notes(raw, midi=dst)

            if stats["removed_duplicates"]:
                logger.info("Удалено дублей нот: %d", stats["removed_duplicates"])
            if stats["merged_overlaps"]:
                logger.info("Склеено перекрывающихся нот одной высоты: %d",
                            stats["merged_overlaps"])

            for pitch, start, end, vel in merged:
                piano.notes.append(pretty_midi.Note(
                    velocity=vel, pitch=pitch, start=start, end=end,
                ))

            unique_pitches = stats["unique_pitches"]
            if unique_pitches < 5:
                logger.warning(
                    "Фортепианная партия подозрительно однообразна: разных высот %d "
                    "на %d нот — возможно, проблема со сведением дорожек",
                    unique_pitches, len(merged),
                )

            dst.instruments.append(piano)
            dst.write(str(out_path))
            logger.info(
                "Фортепианная партия готова",
                extra={"ctx_out": str(out_path), "ctx_notes": len(merged),
                       "ctx_removed_duplicates": stats["removed_duplicates"],
                       "ctx_merged_overlaps": stats["merged_overlaps"],
                       "ctx_unique_pitches": unique_pitches},
            )
        except PianoTranscriberError:
            raise
        except Exception as e:
            logger.exception("Ошибка сведения в фортепиано")
            raise PianoTranscriberError(
                f"Сведение в фортепиано упало: {e}\n"
                f"Промежуточный файл MuScriptor: {tmp_multi}\n"
                "Если ошибка про чтение MIDI — файл повреждён, попробуйте прогнать заново."
            ) from e

        if progress_cb:
            progress_cb(1.0, "Готово")

        return out_path