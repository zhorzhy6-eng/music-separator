"""
Проверки бенчмарка моделей (tools/benchmark.py).

Считаются ли метрики правильно — проверяем на простых случаях, где ответ
очевиден: если «результат разделения» равен эталону, SDR должен быть высоким;
если это шум — низким. Реальные модели здесь не запускаются.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import benchmark  # noqa: E402  (файл лежит в tools/, не в пакете)

SR = 44100


def _tone(freq: float, amp: float = 0.4, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stereo(mono: np.ndarray) -> np.ndarray:
    return np.repeat(mono[:, None], 2, axis=1)


# ============================================================
#  Метрики
# ============================================================
def test_perfect_separation_scores_high():
    """Результат = эталон: SDR высокий, и метрики есть для обоих стемов."""
    vocals = _stereo(_tone(440.0))
    instr = _stereo(_tone(220.0))

    metrics = benchmark.compute_metrics(vocals, instr, vocals, instr)

    assert metrics["sdr_vocals"] > 20
    assert metrics["sdr_instrumental"] > 20
    assert metrics["corr_vocals"] == pytest.approx(1.0, abs=1e-3)
    for key in ("sdr", "sir", "sar"):
        for stem in benchmark.STEMS:
            assert f"{key}_{stem}" in metrics


def test_swapped_stems_score_low():
    """Перепутали вокал и минус — SDR должен провалиться."""
    vocals = _stereo(_tone(440.0))
    instr = _stereo(_tone(220.0))

    metrics = benchmark.compute_metrics(vocals, instr, instr, vocals)

    assert metrics["sdr_vocals"] < 0
    assert metrics["sdr_instrumental"] < 0


def test_different_lengths_are_trimmed():
    """Разная длина эталона и результата не должна ронять расчёт."""
    vocals = _stereo(_tone(440.0, seconds=2.0))
    instr = _stereo(_tone(220.0, seconds=2.0))
    short_v = _stereo(_tone(440.0, seconds=1.5))
    short_i = _stereo(_tone(220.0, seconds=1.5))

    metrics = benchmark.compute_metrics(vocals, instr, short_v, short_i)

    assert "sdr_vocals" in metrics


def test_approx_metrics_matches_expectations():
    """Упрощённые метрики (без mir_eval): идеальная оценка — высоко, шум — низко."""
    ref = np.vstack([_tone(440.0), _tone(220.0)])
    good = ref.copy()
    bad = np.vstack([np.random.default_rng(0).normal(0, 0.2, ref.shape[1]),
                     np.random.default_rng(1).normal(0, 0.2, ref.shape[1])])

    sdr_good, sir_good, sar_good = benchmark._approx_metrics(ref, good)
    sdr_bad, _, _ = benchmark._approx_metrics(ref, bad.astype(np.float32))

    assert len(sdr_good) == len(sir_good) == len(sar_good) == 2
    assert all(value > 20 for value in sdr_good)
    assert all(value < 0 for value in sdr_bad)


def test_metrics_fall_back_without_mir_eval(monkeypatch):
    """Нет mir_eval — считаем упрощённо и честно помечаем это в отчёте."""
    vocals = _stereo(_tone(440.0))
    instr = _stereo(_tone(220.0))

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("mir_eval"):
            raise ImportError("нет mir_eval")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    metrics = benchmark.compute_metrics(vocals, instr, vocals, instr)

    assert metrics["metrics_source"] == "approx"
    assert "note" in metrics
    assert metrics["sdr_vocals"] > 20


# ============================================================
#  Данные и движки
# ============================================================
def _make_track(directory: Path, name: str, complete: bool = True) -> None:
    track = directory / name
    track.mkdir(parents=True, exist_ok=True)
    sf.write(str(track / "mix.wav"), _stereo(_tone(440.0)), SR, subtype="PCM_16")
    if complete:
        sf.write(str(track / "vocals.wav"), _stereo(_tone(440.0)), SR, subtype="PCM_16")
        sf.write(str(track / "instrumental.wav"), _stereo(_tone(220.0)), SR, subtype="PCM_16")


def test_list_tracks_takes_only_complete_folders(tmp_path):
    _make_track(tmp_path, "track_01")
    _make_track(tmp_path, "track_02", complete=False)   # нет эталонов
    (tmp_path / "README.txt").write_text("не папка", encoding="utf-8")

    tracks = benchmark.list_tracks(tmp_path)

    assert [p.name for p in tracks] == ["track_01"]


def test_missing_data_dir_is_not_an_error(tmp_path):
    assert benchmark.list_tracks(tmp_path / "нет-такой-папки") == []


def test_load_audio_resamples_to_base_rate(tmp_path):
    """Файл на другой частоте пересчитывается: иначе метрики бессмысленны."""
    path = tmp_path / "t.wav"
    t = np.arange(int(0.5 * 22050), dtype=np.float64) / 22050
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
             22050, subtype="PCM_16")

    data, sr = benchmark.load_audio(path)

    assert sr == benchmark.BASE_SR
    assert data.shape[0] == pytest.approx(0.5 * benchmark.BASE_SR, rel=0.01)


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)], axis=1)
    assert np.allclose(benchmark.to_mono(stereo), 0.5)


def test_engine_without_model_file_is_skipped():
    """Модели нет на диске — движок пропускаем, а не качаем гигабайты молча."""
    ok, why = benchmark.engine_model_present("bs:BS-Roformer-SW.ckpt")
    if not (benchmark.MODELS_DIR / "BS-Roformer-SW.ckpt").exists():
        assert ok is False
        assert "нет файла" in why

    ok_unknown, why_unknown = benchmark.engine_model_present("bs:такого-нет.ckpt")
    assert ok_unknown is False


def test_demucs_engine_is_considered_available():
    ok, why = benchmark.engine_model_present("demucs:htdemucs_ft")
    assert ok is True and why == ""
