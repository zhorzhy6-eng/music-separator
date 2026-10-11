"""
Транскрипция аудио в MIDI без basic-pitch (Python 3.14-совместимая версия).
Использует torchcrepe для определения высоты тона + pretty_midi для записи.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import torch
import torchcrepe

logger = logging.getLogger(__name__)

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

            if progress_cb:
                progress_cb(0.3, "Определение высоты тона")

            # torchcrepe ждёт tensor (batch, samples)
            audio_t = torch.from_numpy(audio_mono).unsqueeze(0)
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # hop_length=160 при sr=16000 даёт ~100 кадров/сек — приемлемая точность
            f0, confidence = torchcrepe.predict(
                audio_t,
                sample_rate=sr,
                hop_length=160,
                fmin=50.0,
                fmax=2000.0,
                model="full",
                batch_size=512,
                device=device,
                return_periodicity=True,
            )
            f0 = f0.squeeze().cpu().numpy()
            confidence = confidence.squeeze().cpu().numpy()

            if progress_cb:
                progress_cb(0.8, "Сборка MIDI")

            # Строим pretty_midi
            midi = pretty_midi.PrettyMIDI()
            program = GM_INSTRUMENTS.get(instrument_name, 0)
            instrument = pretty_midi.Instrument(program=program)

            frame_dur = 160.0 / sr
            min_conf = 0.3
            min_dur = 0.05  # сек

            notes = self._frames_to_notes(f0, confidence, frame_dur, min_conf, min_dur)
            for start, end, pitch in notes:
                instrument.notes.append(pretty_midi.Note(
                    velocity=90, pitch=int(pitch), start=float(start), end=float(end),
                ))

            midi.instruments.append(instrument)
            midi.write(str(out_path))

            if progress_cb:
                progress_cb(1.0, "Готово")

            logger.info(
                "Транскрипция завершена (torchcrepe)",
                extra={"ctx_in": str(in_path), "ctx_out": str(out_path),
                       "ctx_notes": len(notes)},
            )
            return out_path

        except Exception as e:
            logger.exception("Ошибка транскрипции")
            raise TranscriptionError(f"Ошибка транскрипции: {e}") from e

    @staticmethod
    def _frames_to_notes(f0, conf, frame_dur, min_conf, min_dur):
        """Склеивает кадры F0 в ноты. Возвращает [(start, end, midi_pitch), ...]."""
        notes = []
        cur_pitch = None
        cur_start = None

        def hz_to_midi(hz):
            return 69 + 12 * np.log2(hz / 440.0)

        for i, (freq, c) in enumerate(zip(f0, conf)):
            t = i * frame_dur
            if c < min_conf or freq <= 0 or not np.isfinite(freq):
                # пауза — закрываем текущую ноту
                if cur_pitch is not None:
                    dur = t - cur_start
                    if dur >= min_dur:
                        notes.append((cur_start, t, cur_pitch))
                    cur_pitch = None
                    cur_start = None
                continue

            pitch = hz_to_midi(freq)
            # если питч сильно изменился (>0.5 полутона) — закрываем ноту и начинаем новую
            if cur_pitch is None:
                cur_pitch = pitch
                cur_start = t
            elif abs(pitch - cur_pitch) > 0.5:
                dur = t - cur_start
                if dur >= min_dur:
                    notes.append((cur_start, t, round(cur_pitch)))
                cur_pitch = pitch
                cur_start = t

        # закрыть последнюю
        if cur_pitch is not None:
            t = len(f0) * frame_dur
            if t - cur_start >= min_dur:
                notes.append((cur_start, t, round(cur_pitch)))

        return notes