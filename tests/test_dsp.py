"""
Проверки DSP: фильтры и эквалайзер.

Главное, что здесь ловится: «заявленная глубина notch = фактической».
Раньше параметр означал коэффициент смешивания, и −3.0 дБ в настройке давали
провал −10.7 дБ — то есть фильтр резал втрое сильнее, чем думал человек.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.dsp import (
    TARGET_PEAK,
    frequency_response_db,
    high_pass,
    low_pass,
    normalize_pair,
    pair_peak,
    peaking_eq,
    peaking_eq_sos,
)

SR = 44100


# ---------- peaking EQ ----------
@pytest.mark.parametrize("gain_db", [-1.5, -3.0, -6.0, 3.0])
def test_peaking_eq_depth_matches_spec(gain_db):
    """АЧХ фильтра на центральной частоте равна заявленной глубине."""
    f0, q = 2500.0, 1.5
    sos = peaking_eq_sos(SR, f0, q, gain_db)
    (measured,) = frequency_response_db(sos, SR, [f0])
    assert measured == pytest.approx(gain_db, abs=0.1)


def test_peaking_eq_leaves_other_bands_alone():
    """Далеко от центральной частоты фильтр не трогает сигнал."""
    sos = peaking_eq_sos(SR, 2500.0, 1.5, -3.0)
    low, high = frequency_response_db(sos, SR, [100.0, 15000.0])
    assert abs(low) < 0.5
    assert abs(high) < 1.0


def test_peaking_eq_out_of_range_is_noop():
    """Частота выше половины частоты дискретизации — фильтр не применяется."""
    x = np.ones(1000, dtype=np.float32)
    out = peaking_eq(x, SR, 30000.0, 1.0, -6.0)
    assert np.array_equal(out, x)


def test_peaking_eq_zero_gain_is_noop():
    x = np.ones(1000, dtype=np.float32)
    assert np.array_equal(peaking_eq(x, SR, 2500.0, 1.5, 0.0), x)


# ---------- Фильтры ----------
def _sine(freq: float, sr: int = SR, sec: float = 1.0) -> np.ndarray:
    t = np.arange(int(sec * sr), dtype=np.float64) / sr
    return np.sin(2 * np.pi * freq * t)


def test_high_pass_cuts_below_cutoff():
    """HPF 100 Гц: 40 Гц давится, 1000 Гц проходит."""
    low = high_pass(_sine(40.0), SR, 100.0, 2)
    mid = high_pass(_sine(1000.0), SR, 100.0, 2)
    assert np.sqrt(np.mean(low ** 2)) < 0.2
    assert np.sqrt(np.mean(mid ** 2)) > 0.6


def test_low_pass_cuts_above_cutoff():
    """LPF 19 кГц: 1 кГц проходит, 20 кГц давится."""
    mid = low_pass(_sine(1000.0), SR, 19000.0, 2)
    high = low_pass(_sine(20000.0), SR, 19000.0, 2)
    assert np.sqrt(np.mean(mid ** 2)) > 0.6
    assert np.sqrt(np.mean(high ** 2)) < 0.9


# ---------- Нормализация пары ----------
def test_normalize_pair_single_gain_for_both():
    """Пара нормируется ОДНИМ коэффициентом — взаимный баланс сохраняется."""
    vocals = (0.9 * _sine(440.0)).astype(np.float32)
    instr = (0.3 * _sine(220.0)).astype(np.float32)
    before = float(np.max(np.abs(instr)) / np.max(np.abs(vocals)))

    out = normalize_pair({"vocals": vocals, "instrumental": instr})

    assert pair_peak(out) == pytest.approx(TARGET_PEAK, abs=1e-3)
    after = float(np.max(np.abs(out["instrumental"])) / np.max(np.abs(out["vocals"])))
    assert after == pytest.approx(before, rel=1e-3)


def test_normalize_pair_silence_does_not_crash():
    """Тишина на входе — не делим на ноль."""
    out = normalize_pair({"vocals": np.zeros(100, dtype=np.float32)})
    assert np.allclose(out["vocals"], 0.0)
