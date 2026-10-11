"""
Утилиты для работы с аудиофайлами.
Все функции логируют свои действия и понятно бросают исключения.
"""
from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}

# Частота, на которой работает весь тракт по умолчанию
DEFAULT_SAMPLE_RATE = 44100


class AudioError(Exception):
    """Базовое исключение для ошибок работы с аудио."""


def get_output_sample_rate() -> int:
    """
    Частота дискретизации выходных файлов из настроек (output_sample_rate).

    Некорректное значение не ломает тракт: предупреждаем и берём 44100.
    """
    from core.config import get_config

    try:
        value = int(get_config().output_sample_rate)
    except Exception:
        logger.warning("Не удалось прочитать output_sample_rate — беру %s",
                       DEFAULT_SAMPLE_RATE)
        return DEFAULT_SAMPLE_RATE
    if not (8000 <= value <= 192000):
        logger.warning("output_sample_rate=%s вне диапазона 8000..192000 — беру %s",
                       value, DEFAULT_SAMPLE_RATE)
        return DEFAULT_SAMPLE_RATE
    return value


def resample_audio(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """
    Пересчитывает сигнал к target_sr. Ось времени — первая (samples, ...).

    Нужна, когда частота вывода в настройках отличается от частоты модели.
    """
    if sr == target_sr or y.size == 0:
        return y
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr), int(target_sr))
    up, down = target_sr // g, sr // g
    logger.info("Ресемплинг вывода: %s → %s Гц (up=%s, down=%s)",
                sr, target_sr, up, down)
    return resample_poly(y, up, down, axis=0).astype(np.float32)


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def get_audio_info(path: str | Path) -> dict[str, Any]:
    """
    Возвращает базовую информацию о файле.
    Для MP3/M4A — пытается прочитать через soundfile,
    если не получается — через ffprobe (FFmpeg).
    """
    p = Path(path)
    if not p.exists():
        raise AudioError(f"Файл не найден: {p}")
    if not is_supported(p):
        raise AudioError(f"Неподдерживаемый формат: {p.suffix}")

    # Сначала пробуем soundfile (работает для WAV/FLAC/OGG)
    try:
        info = sf.info(str(p))
        return {
            "name": p.name,
            "path": str(p.resolve()),
            "duration_sec": round(info.duration, 2),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": p.suffix.lower().lstrip("."),
            "bitrate_kbps": None,
        }
    except Exception:
        pass

    # Для MP3/M4A — через ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels,duration,bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=0",
                str(p),
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            raise AudioError(
                f"FFprobe не смог прочитать файл.\n"
                f"stderr: {result.stderr.strip()}\n"
                f"Скорее всего, файл повреждён или FFmpeg не установлен."
            )
        data = {}
        for line in result.stdout.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
        return {
            "name": p.name,
            "path": str(p.resolve()),
            "duration_sec": round(float(data.get("duration", 0)), 2),
            "sample_rate": int(data.get("sample_rate", 44100)),
            "channels": int(data.get("channels", 2)),
            "format": p.suffix.lower().lstrip("."),
            "bitrate_kbps": int(data.get("bit_rate", 0)) // 1000 or None,
        }
    except FileNotFoundError:
        raise AudioError(
            "FFmpeg не установлен. Установите его командой:\n"
            "    winget install Gyan.FFmpeg\n"
            "и перезапустите PowerShell."
        )
    except AudioError:
        raise
    except Exception as e:
        raise AudioError(f"Ошибка чтения файла: {e}") from e


def load_mono(path: str | Path, sr: int | None = None, max_sec: float | None = None):
    """
    Загружает аудио в моно (через soundfile, для WAV/FLAC/OGG).

    sr — если задан и отличается от частоты файла, сигнал пересчитывается.
    Раньше параметр молча игнорировался: функция всегда возвращала родную частоту.
    """
    y, real_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)

    if max_sec is not None:
        limit = int(max_sec * real_sr)
        if 0 < limit < len(y):
            y = y[:limit]

    if sr is not None and sr != real_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(real_sr), int(sr))
        y = resample_poly(y, sr // g, real_sr // g).astype(np.float32)
        logger.info("load_mono: ресемплинг %s → %s Гц", real_sr, sr)
        real_sr = sr

    return y, real_sr


def write_wav(path: str | Path, y, sr: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), y, sr, subtype="PCM_16")
    logger.info("Записан WAV", extra={"ctx_file": str(p)})


def measure_notch_depth(
    sr: int, center_hz: float, q: float, gain_db: float, duration_sec: float = 1.0
) -> float:
    """
    Измеряет ФАКТИЧЕСКОЕ ослабление peaking-EQ на центральной частоте.

    Сейчас фильтр строится по RBJ biquad (core/dsp.py::peaking_eq_sos), где gain_db —
    честная глубина, поэтому измерение лишь подтверждает заданное значение.
    Функция оставлена как самопроверка: если формулу фильтра однажды поменяют,
    расхождение между заявленным и измеренным будет видно в логе.

    Возвращает: ослабление в дБ (отрицательное число).
    Бросает AudioError, если scipy недоступен или частота некорректна.
    """
    try:
        from scipy.signal import sosfilt
    except ImportError as e:
        raise AudioError("scipy не установлен — измерение notch невозможно") from e

    nyq = 0.5 * sr
    if center_hz <= 0 or center_hz >= nyq:
        raise AudioError(f"Некорректная частота notch: {center_hz} Гц при sr={sr}")

    # Формула фильтра берётся из общего модуля, чтобы не разъезжаться с продом
    from core.dsp import peaking_eq_sos

    n = max(4096, int(duration_sec * sr))
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.sin(2.0 * math.pi * center_hz * t)

    sos = peaking_eq_sos(sr, center_hz, q, gain_db)
    out = sosfilt(sos, sig)

    skip = n // 4  # отбрасываем переходный процесс фильтра
    rms_in = float(np.sqrt(np.mean(sig[skip:] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[skip:] ** 2)))
    if rms_in <= 1e-12:
        return 0.0
    return 20.0 * math.log10(max(rms_out, 1e-12) / rms_in)


def ensure_wav(path: str | Path, target_sr: int = 44100) -> Path:
    """
    Возвращает путь к WAV-файлу. Если исходник уже WAV — возвращает его.
    Если MP3/M4A/OGG — перекодирует через FFmpeg во временный WAV.

    Это решает проблему, когда Demucs не может прочитать битый MP3.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return p

    tmp_dir = Path(tempfile.gettempdir()) / "ast_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / (p.stem + ".wav")

    logger.info("Перекодирую в WAV через FFmpeg", extra={"ctx_src": str(p)})
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(p),
                "-ar", str(target_sr),
                "-ac", "2",
                "-c:a", "pcm_s16le",
                str(out),
            ],
            capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        raise AudioError(
            "FFmpeg не установлен. Установите его командой:\n"
            "    winget install Gyan.FFmpeg"
        )

    if result.returncode != 0 or not out.exists():
        raise AudioError(
            f"FFmpeg не смог перекодировать файл.\n"
            f"stderr: {result.stderr.strip()[:500]}"
        )

    logger.info("Перекодирование завершено", extra={"ctx_out": str(out)})
    return out