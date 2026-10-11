"""
Общие фикстуры тестов.

Правило: тесты НЕ трогают пользовательские данные (output/, logs/analysis,
config/settings.json). Всё, что нужно, создаётся во временном каталоге pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Корень проекта в путь импорта — тесты запускаются из любой папки
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.stems import StemSet  # noqa: E402

SR = 44100


@pytest.fixture
def tmp_audio(tmp_path):
    """Фабрика WAV-файлов во временном каталоге.

    Использование:  path = tmp_audio(name="tone.wav", freq=440.0)
    """
    def make(name: str = "test.wav", freq: float = 440.0, sec: float = 1.0,
             sr: int = SR, amp: float = 0.5, channels: int = 1) -> Path:
        n = int(sec * sr)
        t = np.arange(n, dtype=np.float64) / sr
        mono = amp * np.sin(2.0 * np.pi * freq * t)
        data = mono if channels == 1 else np.repeat(mono[:, None], channels, axis=1)
        path = tmp_path / name
        sf.write(str(path), data.astype(np.float32), sr, subtype="PCM_16")
        return path

    return make


@pytest.fixture
def silent_wav(tmp_path):
    """Один короткий WAV — когда содержание не важно."""
    path = tmp_path / "silence.wav"
    sf.write(str(path), np.zeros(SR // 2, dtype=np.float32), SR, subtype="PCM_16")
    return path


@pytest.fixture
def fake_stems(tmp_path) -> StemSet:
    """Синтетический StemSet: файлы существуют, но пустые."""
    vocals = tmp_path / "track_vocals.wav"
    instr = tmp_path / "track_instrumental.wav"
    for p in (vocals, instr):
        sf.write(str(p), np.zeros(1024, dtype=np.float32), SR, subtype="PCM_16")
    return StemSet(vocals=vocals, instrumental=instr)


@pytest.fixture
def pair_wavs(tmp_path):
    """Пара вокал+минус: вокал громче, минус тише (для тестов нормализации)."""
    vocals = tmp_path / "pair_vocals.wav"
    instr = tmp_path / "pair_instrumental.wav"
    t = np.arange(SR, dtype=np.float64) / SR
    sf.write(str(vocals), (1.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
             SR, subtype="FLOAT")
    sf.write(str(instr), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
             SR, subtype="FLOAT")
    return vocals, instr
