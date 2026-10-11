"""
Плеер AST с поддержкой pause/resume/seek.
Использует sounddevice.OutputStream с callback.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self) -> None:
        self._y: np.ndarray | None = None
        self._sr: int = 44100
        self._pos: int = 0
        self._volume: float = 0.8
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()
        self._playing = False
        self._current_path: Path | None = None

    # ---------- публичное API ----------
    def load(self, path: str | Path) -> None:
        self.stop()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {p}")
        try:
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
        except Exception:
            logger.exception("Ошибка чтения аудио в плеер")
            raise
        with self._lock:
            self._y = y
            self._sr = sr
            self._pos = 0
            self._current_path = p
        logger.info("Плеер загрузил файл", extra={"ctx_file": str(p)})

    def play(self) -> None:
        with self._lock:
            if self._y is None:
                logger.warning("Play без загруженного файла")
                return
            if self._pos >= len(self._y):
                self._pos = 0

        if self._playing:
            return

        self._open_stream()
        self._playing = True
        logger.info("Воспроизведение начато",
                    extra={"ctx_file": str(self._current_path), "ctx_pos": self._pos})

    def pause(self) -> None:
        self._close_stream()
        self._playing = False
        logger.info("Пауза", extra={"ctx_pos": self._pos})

    def stop(self) -> None:
        self._close_stream()
        self._playing = False
        with self._lock:
            self._pos = 0
        logger.info("Стоп")

    def seek(self, seconds: float) -> None:
        """Перемотка. Если играет — перезапускаем поток с новой позиции."""
        with self._lock:
            if self._y is None:
                return
            dur = len(self._y) / self._sr
            sec = max(0.0, min(float(seconds), dur))
            self._pos = int(sec * self._sr)
            was_playing = self._playing
        logger.info("Перемотка", extra={"ctx_pos_sec": round(sec, 2)})

        if was_playing:
            self._close_stream()
            self._open_stream()

    def set_volume(self, v: float) -> None:
        self._volume = max(0.0, min(1.0, float(v)))

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def current_file(self) -> Path | None:
        return self._current_path

    @property
    def position(self) -> float:
        with self._lock:
            if self._y is None or self._sr == 0:
                return 0.0
            pos_sec = self._pos / self._sr
            dur = len(self._y) / self._sr
            return max(0.0, min(pos_sec, dur))

    @property
    def duration(self) -> float:
        with self._lock:
            if self._y is None or self._sr == 0:
                return 0.0
            return len(self._y) / self._sr

    # ---------- внутреннее ----------
    def _open_stream(self) -> None:
        self._close_stream()
        with self._lock:
            if self._y is None:
                return
            y = self._y
            sr = self._sr
        channels = 1 if y.ndim == 1 else y.shape[1]

        try:
            self._stream = sd.OutputStream(
                samplerate=sr,
                channels=channels,
                dtype="float32",
                callback=self._callback,
                finished_callback=self._on_stream_finished,
                blocksize=2048,
            )
            self._stream.start()
        except Exception:
            logger.exception("Не удалось открыть аудиопоток")
            self._stream = None
            self._playing = False
            raise

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Ошибка закрытия потока")
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        if status:
            logger.warning("Stream status: %s", status)
        with self._lock:
            if self._y is None:
                outdata[:] = 0
                raise sd.CallbackStop
            remaining = len(self._y) - self._pos
            if remaining <= 0:
                outdata[:] = 0
                raise sd.CallbackStop
            n = min(frames, remaining)
            chunk = self._y[self._pos:self._pos + n] * self._volume
            self._pos += n

        if chunk.ndim == 1 and outdata.ndim == 2:
            outdata[:n, 0] = chunk
        else:
            outdata[:n] = chunk
        if n < frames:
            outdata[n:] = 0

    def _on_stream_finished(self):
        with self._lock:
            at_end = self._y is not None and self._pos >= len(self._y)
        if at_end:
            self._playing = False
            logger.info("Воспроизведение завершено")