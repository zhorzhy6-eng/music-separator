"""
Анализ аудиофайла: уровни, спектр, стерео, тишина, F0.
Сохраняет отчёт (.json, .txt) и графики (.png) рядом с файлом.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import matplotlib
# force=True: анализатор рисует графики в РАБОЧЕМ потоке, поэтому бэкенд обязан
# быть безголовым. Иначе UI-модуль (ui/components.py) переключит matplotlib на
# TkAgg, и попытка рисовать из фонового потока полезет в Tk — с предупреждением
# «Starting a Matplotlib GUI outside of the main thread will likely fail»
# и риском уронить процесс.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from core.utils import to_db

logger = logging.getLogger(__name__)

# Октавные полосы (Гц) — центр каждой
OCTAVE_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


_to_db = to_db  # историческое имя: приватная функция, но менять вызовы не будем


class AnalysisError(Exception):
    """Ошибка анализа аудио."""


def analyze_audio(
    audio_path: str | Path,
    report_dir: str | Path | None = None,
    progress_cb=None,
) -> dict:
    """
    Анализирует аудио и сохраняет отчёт.

    Возвращает dict со всеми метриками. Также создаёт:
      - <report_dir>/<name>_analysis.json
      - <report_dir>/<name>_analysis.txt
      - <report_dir>/<name>_waveform.png
      - <report_dir>/<name>_spectrogram.png
    """
    p = Path(audio_path)
    if not p.exists():
        raise AnalysisError(f"Файл не найден: {p}")

    if report_dir is None:
        report_dir = p.parent / "analysis"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    base = p.stem
    if progress_cb:
        progress_cb(0.05, "Чтение аудио")

    # ---------- Чтение ----------
    try:
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception as e:
        logger.exception("Не удалось прочитать файл для анализа")
        raise AnalysisError(f"Ошибка чтения: {e}") from e

    n_samples, n_channels = y.shape
    duration = n_samples / sr
    info = sf.info(str(p))

    # моно для спектра
    mono = y.mean(axis=1)

    if progress_cb:
        progress_cb(0.20, "Уровни и клиппинг")

    # ---------- Уровни ----------
    peak = float(np.max(np.abs(y)))
    peak_db = _to_db(peak)

    # RMS по всему треку
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = _to_db(rms)

    # Приблизительный LUFS (не ITU-R BS.1770, но близко по громкости)
    lufs_approx = rms_db - 3.0  # эмпирическая поправка

    # Клиппинг: сколько сэмплов ≥ 0.99
    clipping_samples = int(np.sum(np.abs(y) >= 0.99))
    clipping_pct = 100.0 * clipping_samples / y.size if y.size else 0.0

    # ---------- Тишина ----------
    if progress_cb:
        progress_cb(0.35, "Анализ тишины")

    frame_len = int(0.05 * sr)  # 50 мс
    if frame_len < 1:
        frame_len = 1
    n_frames = n_samples // frame_len
    if n_frames > 0:
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        frame_db = 20 * np.log10(np.maximum(frame_rms, 1e-10))
        silence_frames = int(np.sum(frame_db < -60))
        silence_pct = 100.0 * silence_frames / n_frames
    else:
        silence_pct = 0.0

    # ---------- Спектр по октавам ----------
    if progress_cb:
        progress_cb(0.50, "Спектр")

    # FFT по всему треку (усреднённо)
    spec = np.abs(np.fft.rfft(mono[: min(len(mono), sr * 60)]))  # первые 60 сек
    freqs = np.fft.rfftfreq(min(len(mono), sr * 60), d=1.0 / sr)
    spec_db = 20 * np.log10(np.maximum(spec, 1e-10))

    octave_energy = {}
    for center in OCTAVE_BANDS:
        lo = center / math.sqrt(2)
        hi = center * math.sqrt(2)
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            octave_energy[center] = float(np.mean(spec_db[mask]))
        else:
            octave_energy[center] = float("-inf")

    # ---------- Стерео ----------
    if progress_cb:
        progress_cb(0.65, "Стерео")

    stereo = {}
    if n_channels >= 2:
        left = y[:, 0]
        right = y[:, 1]
        # корреляция
        if np.std(left) > 1e-9 and np.std(right) > 1e-9:
            corr = float(np.corrcoef(left, right)[0, 1])
        else:
            corr = 0.0
        # ширина стерео = RMS разницы
        side = (left - right) / 2.0
        mid = (left + right) / 2.0
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        mid_rms = float(np.sqrt(np.mean(mid ** 2)))
        width = side_rms / mid_rms if mid_rms > 1e-9 else 0.0
        stereo = {
            "correlation": round(corr, 4),
            "width": round(float(width), 4),
        }
    else:
        stereo = {"correlation": None, "width": None}

    # ---------- F0 (простая оценка через автокорреляцию) ----------
    if progress_cb:
        progress_cb(0.80, "Оценка F0")

    f0_estimate = _estimate_f0(mono, sr)

    # ---------- Графики ----------
    if progress_cb:
        progress_cb(0.90, "Графики")

    wave_png = report_dir / f"{base}_waveform.png"
    spec_png = report_dir / f"{base}_spectrogram.png"
    _save_waveform(mono, sr, wave_png)
    _save_spectrogram(mono, sr, spec_png)

    # ---------- Отчёт ----------
    report = {
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "file": str(p.resolve()),
        "name": p.name,
        "format": info.format,
        "subtype": info.subtype,
        "duration_sec": round(duration, 2),
        "sample_rate": int(sr),
        "channels": int(n_channels),
        "samples": int(n_samples),
        "levels": {
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2),
            "lufs_approx": round(lufs_approx, 2),
            "clipping_samples": clipping_samples,
            "clipping_pct": round(clipping_pct, 4),
        },
        "silence": {
            "threshold_db": -60,
            "silence_pct": round(silence_pct, 2),
        },
        "octave_energy_db": {str(k): round(v, 2) for k, v in octave_energy.items()},
        "stereo": stereo,
        "f0_hz": f0_estimate,
        "artifacts": {
            "waveform_png": str(wave_png),
            "spectrogram_png": str(spec_png),
        },
    }

    json_path = report_dir / f"{base}_analysis.json"
    txt_path = report_dir / f"{base}_analysis.txt"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    txt_path.write_text(_render_text_report(report), encoding="utf-8")

    if progress_cb:
        progress_cb(1.0, "Готово")

    logger.info(
        "Анализ завершён",
        extra={
            "ctx_file": str(p),
            "ctx_json": str(json_path),
            "ctx_rms_db": rms_db,
            "ctx_peak_db": peak_db,
        },
    )
    return report


# ============================================================
#  Внутренние помощники
# ============================================================

def _estimate_f0(mono: np.ndarray, sr: int) -> dict:
    """Грубая оценка основного тона через автокорреляцию по кадрам 100 мс."""
    frame = int(0.1 * sr)
    if frame <= 0 or len(mono) < frame * 3:
        return {"median_hz": None, "min_hz": None, "max_hz": None}

    # берём окно 30 сек из середины
    mid = len(mono) // 2
    start = max(0, mid - 15 * sr)
    end = min(len(mono), mid + 15 * sr)
    seg = mono[start:end]

    f0_list = []
    hop = frame // 2
    for i in range(0, len(seg) - frame, hop):
        chunk = seg[i:i + frame]
        if np.sqrt(np.mean(chunk ** 2)) < 1e-3:
            continue  # тишина — пропускаем
        # автокорреляция
        chunk = chunk - chunk.mean()
        ac = np.correlate(chunk, chunk, mode="full")[len(chunk) - 1:]
        # ищем первый значимый пик в диапазоне 60..1000 Гц
        min_lag = int(sr / 1000)
        max_lag = int(sr / 60)
        if max_lag >= len(ac):
            continue
        region = ac[min_lag:max_lag]
        if len(region) == 0:
            continue
        peak_idx = int(np.argmax(region)) + min_lag
        if ac[0] > 0 and ac[peak_idx] / ac[0] > 0.3:
            f0 = sr / peak_idx
            if 60 <= f0 <= 1000:
                f0_list.append(f0)

    if not f0_list:
        return {"median_hz": None, "min_hz": None, "max_hz": None}

    arr = np.array(f0_list)
    return {
        "median_hz": round(float(np.median(arr)), 2),
        "min_hz": round(float(np.percentile(arr, 5)), 2),
        "max_hz": round(float(np.percentile(arr, 95)), 2),
    }


def _save_waveform(mono: np.ndarray, sr: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.5), dpi=100)
    step = max(1, len(mono) // 8000)
    t = np.arange(0, len(mono), step) / sr
    ax.plot(t, mono[::step], linewidth=0.4, color="#3b8ed0")
    ax.set_xlabel("сек")
    ax.set_ylabel("amplitude")
    ax.set_title("Waveform")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_spectrogram(mono: np.ndarray, sr: int, out_path: Path) -> None:
    """Спектрограмма через matplotlib specgram (без librosa)."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    # ограничим 60 сек для скорости
    seg = mono[: sr * 60]
    ax.specgram(seg, NFFT=2048, Fs=sr, noverlap=1024, cmap="magma")
    ax.set_xlabel("сек")
    ax.set_ylabel("Гц")
    ax.set_ylim(0, 8000)
    ax.set_title("Spectrogram (0–8 кГц)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _render_text_report(r: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"АУДИО-АНАЛИЗ: {r['name']}")
    lines.append("=" * 60)
    lines.append(f"Время анализа:  {r['analyzed_at']}")
    lines.append(f"Файл:           {r['file']}")
    lines.append(f"Формат:         {r['format']} / {r['subtype']}")
    lines.append(f"Длительность:   {r['duration_sec']} сек")
    lines.append(f"Sample rate:    {r['sample_rate']} Гц")
    lines.append(f"Каналов:        {r['channels']}")
    lines.append("")
    lines.append("--- УРОВНИ ---")
    lv = r["levels"]
    lines.append(f"Peak:           {lv['peak_db']} dBFS")
    lines.append(f"RMS:            {lv['rms_db']} dBFS")
    lines.append(f"LUFS (прибл.):  {lv['lufs_approx']}")
    lines.append(f"Клиппинг:       {lv['clipping_samples']} сэмплов ({lv['clipping_pct']}%)")
    lines.append("")
    lines.append("--- ТИШИНА ---")
    lines.append(f"Порог:          {r['silence']['threshold_db']} dB")
    lines.append(f"Доля тишины:    {r['silence']['silence_pct']}%")
    lines.append("")
    lines.append("--- СПЕКТР (средняя энергия по октавам, dB) ---")
    for k, v in r["octave_energy_db"].items():
        lines.append(f"  {k:>5} Гц: {v}")
    lines.append("")
    lines.append("--- СТЕРЕО ---")
    st = r["stereo"]
    lines.append(f"Корреляция L/R: {st['correlation']}")
    lines.append(f"Ширина стерео:  {st['width']}")
    lines.append("")
    lines.append("--- ОСНОВНОЙ ТОН (F0) ---")
    f0 = r["f0_hz"]
    lines.append(f"Медиана:        {f0['median_hz']} Гц")
    lines.append(f"5-й перцентиль: {f0['min_hz']} Гц")
    lines.append(f"95-й перцентиль:{f0['max_hz']} Гц")
    lines.append("")
    lines.append("--- АРТЕФАКТЫ ---")
    for k, v in r["artifacts"].items():
        lines.append(f"{k}: {v}")
    lines.append("=" * 60)
    return "\n".join(lines)