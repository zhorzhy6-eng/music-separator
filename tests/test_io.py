"""
Проверки ввода-вывода аудио: пик выше 1.0 не должен превращаться в клиппинг.

Раньше перегруз лечился np.clip — плоские верхушки в самых громких местах.
Правильное решение: опустить ВЕСЬ сигнал одним коэффициентом, сохранив форму.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from core.audio_utils import load_mono, write_wav
from core.dsp import pair_peak, prevent_clipping_pair

SR = 44100


def test_load_mono_returns_mono(tmp_path):
    """Стерео читается как один канал."""
    stereo = np.stack([np.sin(np.arange(SR) / 100.0)] * 2, axis=1).astype(np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, SR, subtype="PCM_16")

    y, sr = load_mono(path)
    assert sr == SR
    assert y.ndim == 1
    assert len(y) == SR


def test_load_mono_resamples(tmp_path):
    """Просьба о другой частоте действительно пересчитывает сигнал."""
    y0 = np.sin(2 * np.pi * 440 * np.arange(SR, dtype=np.float64) / SR)
    path = tmp_path / "tone.wav"
    sf.write(str(path), y0.astype(np.float32), SR, subtype="PCM_16")

    y, sr = load_mono(path, sr=22050)
    assert sr == 22050
    assert abs(len(y) - SR // 2) <= 2


def test_write_wav_creates_file(tmp_path):
    path = tmp_path / "out" / "written.wav"
    write_wav(path, np.zeros(100, dtype=np.float32), SR)
    assert path.exists()
    info = sf.info(str(path))
    assert info.samplerate == SR


def test_loud_peak_is_scaled_not_clipped():
    """Пик 1.5 не клипуется: сигнал опускается одним коэффициентом."""
    t = np.arange(SR, dtype=np.float64) / SR
    loud = (1.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    quiet = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    out = prevent_clipping_pair({"vocals": loud, "instrumental": quiet})

    # Ни одного сэмпла на потолке — то есть клиппинга нет
    assert np.max(np.abs(out["vocals"])) <= 1.0
    assert not np.any(np.abs(out["vocals"]) >= 1.0)
    # Форма сохранена: там, где было тихо, осталось тихо (пропорция не изменилась)
    ratio_before = np.max(np.abs(quiet)) / np.max(np.abs(loud))
    ratio_after = (np.max(np.abs(out["instrumental"]))
                   / np.max(np.abs(out["vocals"])))
    assert abs(ratio_before - ratio_after) < 1e-3


def test_save_and_reload_keeps_peak(tmp_path):
    """Записанный файл читается назад с тем же пиком (в пределах шага PCM_16)."""
    t = np.arange(SR, dtype=np.float64) / SR
    signal = (0.891 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "peak.wav"
    sf.write(str(path), signal, SR, subtype="PCM_16")

    back, _ = sf.read(str(path), dtype="float32")
    assert pair_peak({"x": back}) <= 0.891 + 1e-3
