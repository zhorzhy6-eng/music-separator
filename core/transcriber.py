"""
Транскрипция аудио в MIDI через torchcrepe + pretty_midi.

Важно про тайминги: torchcrepe-модель рассчитана на 16 кГц. Если подать файл
с другой частотой, torchcrepe сам пересчитывает hop_length, и расчёт длительности
кадра через 160/sr начинает систематически врать. Поэтому вход явно
пересчитывается в 16 кГц, а frame_dur считается как hop_length / sr.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import torch
import torchcrepe

logger = logging.getLogger(__name__)

# Частота, под которую обучена модель torchcrepe
TARGET_SR = 16000
# hop_length при 16 кГц → ~100 кадров в секунду
HOP_LENGTH = 160
# Окно медианного сглаживания F0 в кадрах (нечётное)
F0_MEDIAN_WINDOW = 5
# Минимальная длина ноты: короче — это уже не нота, а дрожь трекера
MIN_NOTE_DUR_SEC = 0.08

# Сколько нот в секунду ждём от нормальной мелодии. Если получилось заметно
# меньше — скорее всего материал полифонический, а torchcrepe умеет держать
# только одну высоту за раз (на миксе он почти не набирает уверенность).
EXPECTED_NOTES_PER_SEC = 0.5

# Зазор между нотами одной высоты, который считается «перещёлкиванием» трекера,
# а не новой нотой. Такие пары склеиваем в одну ноту.
NOTE_MERGE_GAP_SEC = 0.015


def _merge_tiny_gaps(
    notes: list[tuple[float, float, int]],
    max_gap: float = NOTE_MERGE_GAP_SEC,
) -> tuple[list[tuple[float, float, int]], int]:
    """
    Склеивает соседние ноты одной высоты, если между ними почти нет паузы.

    На вибрато трекер закрывает и тут же снова открывает ту же высоту: на слух
    это «перещёлкивание» (в QA-прогоне таких серий было 7, это 24 ноты из 329).
    Возвращает (ноты, сколько склеек сделано).
    """
    if not notes:
        return notes, 0

    ordered = sorted(notes, key=lambda n: n[0])
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1], ordered[0][2]]]
    glued = 0
    for start, end, pitch in ordered[1:]:
        last = merged[-1]
        gap = start - last[1]
        if int(pitch) == int(last[2]) and 0.0 <= gap < max_gap:
            last[1] = max(last[1], end)
            glued += 1
        else:
            merged.append([start, end, pitch])

    return [(float(s), float(e), int(p)) for s, e, p in merged], glued

GM_INSTRUMENTS = {
    "Фортепиано": 0,
    "Гитара (нейлон)": 24,
    "Скрипка": 40,
    "Флейта": 73,
    "Саксофон": 65,
    "Синт-лид": 80,
}


class TranscriptionError(Exception):
    """Ошибка транскрипции."""


def _resample_to_target(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """
    Пересчитывает моно-сигнал к target_sr.
    Сначала пробуем librosa, если её нет — scipy.signal.resample_poly.
    """
    if sr == target_sr:
        return y

    try:
        import librosa
        logger.info("Ресемплинг через librosa: %s → %s Гц", sr, target_sr)
        return librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    except ImportError:
        pass

    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target_sr))
        up = target_sr // g
        down = sr // g
        logger.info("Ресемплинг через scipy: %s → %s Гц (up=%s, down=%s)",
                    sr, target_sr, up, down)
        return resample_poly(y, up, down).astype(np.float32)
    except ImportError as e:
        raise TranscriptionError(
            "Нужен ресемплинг к 16 кГц, но ни librosa, ни scipy не установлены"
        ) from e


class MidiTranscriber:
    """
    Транскрипция монофонического/полифонического аудио в MIDI.
    Алгоритм: torchcrepe даёт F0 (частоту) по кадрам → склеиваем в ноты.
    """

    def transcribe(
        self,
        input_audio: str | Path,
        output_midi: str | Path,
        instrument_name: str = "Фортепиано",
        progress_cb=None,
    ) -> Path:
        in_path = Path(input_audio)
        out_path = Path(output_midi)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise TranscriptionError(f"Аудиофайл не найден: {in_path}")

        try:
            if progress_cb:
                progress_cb(0.1, "Чтение аудио")

            # soundfile: (samples, channels) → моно
            audio, sr = sf.read(str(in_path), dtype="float32", always_2d=True)
            audio_mono = audio.mean(axis=1)

            # torchcrepe обучен на 16 кГц — приводим вход явно,
            # иначе модель сама ресемплит и пересчитывает hop_length
            audio_mono = _resample_to_target(audio_mono, sr, TARGET_SR)
            sr = TARGET_SR

            if progress_cb:
                progress_cb(0.3, "Определение высоты тона")

            # torchcrepe ждёт tensor (batch, samples)
            audio_t = torch.from_numpy(np.ascontiguousarray(audio_mono)).unsqueeze(0)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            f0, confidence = torchcrepe.predict(
                audio_t,
                sample_rate=sr,
                hop_length=HOP_LENGTH,
                fmin=50.0,
                fmax=2000.0,
                model="full",
                batch_size=512,
                device=device,
                return_periodicity=True,
            )
            f0 = f0.squeeze().cpu().numpy()
            confidence = confidence.squeeze().cpu().numpy()

            # На очень коротком аудио squeeze() может вернуть 0-мерный массив
            if f0.ndim == 0:
                f0 = f0.reshape(1)
            if confidence.ndim == 0:
                confidence = confidence.reshape(1)

            if progress_cb:
                progress_cb(0.8, "Сборка MIDI")

            # Строим pretty_midi
            midi = pretty_midi.PrettyMIDI()
            program = GM_INSTRUMENTS.get(instrument_name, 0)
            instrument = pretty_midi.Instrument(program=program)

            # Длительность кадра считаем из фактических параметров,
            # а не из магического «160/sr»
            frame_dur = HOP_LENGTH / sr
            min_conf = 0.3

            notes = self._frames_to_notes(f0, confidence, frame_dur, min_conf)
            notes, glued = _merge_tiny_gaps(notes)
            if glued:
                logger.info("Склеено нот: %d (перещёлкивание трекера на одной высоте)",
                            glued)
            for start, end, pitch in notes:
                instrument.notes.append(pretty_midi.Note(
                    velocity=90, pitch=int(pitch), start=float(start), end=float(end),
                ))

            midi.instruments.append(instrument)
            midi.write(str(out_path))

            if progress_cb:
                progress_cb(1.0, "Готово")

            # Проверяем, похоже ли это на транскрипцию вообще.
            # torchcrepe — монофонический трекер: на полноценном миксе он отдаёт
            # единицы нот, и человек ждёт результат, которого не будет.
            dur_sec = len(audio_mono) / sr if sr else 0.0
            expected = dur_sec * EXPECTED_NOTES_PER_SEC
            if expected > 0 and len(notes) < expected:
                logger.warning(
                    "Транскрипция дала мало нот: %d из ~%d ожидаемых (%.0f%% на %.0f с). "
                    "Похоже, материал полифонический — torchcrepe монофонический. "
                    "Для полифонии используйте «🎹 Минус → Пианино» (MuScriptor).",
                    len(notes), int(expected),
                    100.0 * len(notes) / expected, dur_sec,
                )

            logger.info(
                "Транскрипция завершена (torchcrepe)",
                extra={"ctx_in": str(in_path), "ctx_out": str(out_path),
                       "ctx_notes": len(notes), "ctx_frame_dur": round(frame_dur, 6),
                       "ctx_glued_notes": glued,
                       "ctx_expected_notes": int(expected)},
            )
            return out_path

        except TranscriptionError:
            raise
        except Exception as e:
            logger.exception("Ошибка транскрипции")
            raise TranscriptionError(f"Ошибка транскрипции: {e}") from e

    # ---------- кадры F0 → ноты ----------
    @staticmethod
    def _hz_to_midi(hz: float) -> float:
        return 69.0 + 12.0 * np.log2(hz / 440.0)

    @classmethod
    def _smooth_f0(cls, f0: np.ndarray, conf: np.ndarray, min_conf: float,
                   window: int = F0_MEDIAN_WINDOW) -> np.ndarray:
        """
        Медианное сглаживание F0 в логарифмической (MIDI) шкале.

        Без него на вибрато трекер даёт рваные ноты: питч прыгает вокруг
        истинного значения и каждый прыжок закрывает одну ноту и открывает новую.
        Неозвученные кадры (тишина, низкая уверенность) становятся NaN и сглаживанию
        не мешают.
        """
        pitch = np.full(len(f0), np.nan, dtype=np.float64)
        voiced = (conf >= min_conf) & np.isfinite(f0) & (f0 > 0)
        pitch[voiced] = cls._hz_to_midi(f0[voiced])

        if not np.any(voiced) or window < 3:
            return pitch

        half = window // 2
        smoothed = pitch.copy()
        with warnings.catch_warnings():
            # окно может целиком попасть в паузу — это нормально, не спамим в лог
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for i in range(len(pitch)):
                if not voiced[i]:
                    continue
                lo = max(0, i - half)
                hi = min(len(pitch), i + half + 1)
                win = pitch[lo:hi]
                if np.all(np.isnan(win)):
                    continue
                smoothed[i] = np.nanmedian(win)
        return smoothed

    @classmethod
    def _frames_to_notes(cls, f0, conf, frame_dur, min_conf,
                         min_dur: float = MIN_NOTE_DUR_SEC):
        """Склеивает кадры F0 в ноты. Возвращает [(start, end, midi_pitch), ...]."""
        notes = []
        if len(f0) == 0:
            return notes

        pitch = cls._smooth_f0(f0, conf, min_conf)

        cur_pitch = None
        cur_start = None

        for i, value in enumerate(pitch):
            t = i * frame_dur
            if np.isnan(value):
                # пауза — закрываем текущую ноту
                if cur_pitch is not None:
                    if t - cur_start >= min_dur:
                        notes.append((cur_start, t, int(round(cur_pitch))))
                    cur_pitch = None
                    cur_start = None
                continue

            # если питч сильно изменился (>0.5 полутона) — закрываем ноту и начинаем новую
            if cur_pitch is None:
                cur_pitch = value
                cur_start = t
            elif abs(value - cur_pitch) > 0.5:
                if t - cur_start >= min_dur:
                    notes.append((cur_start, t, int(round(cur_pitch))))
                cur_pitch = value
                cur_start = t

        # закрыть последнюю
        if cur_pitch is not None:
            t = len(pitch) * frame_dur
            if t - cur_start >= min_dur:
                notes.append((cur_start, t, int(round(cur_pitch))))

        return notes
