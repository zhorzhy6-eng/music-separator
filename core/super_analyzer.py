"""
СУПЕР-АНАЛИЗ ПАРЫ: вокал + минус (v4.1).

Пять уровней + фокус на:
  - Phase Smearing (смазывание атаки)
  - Spectral Tilt (приглушение верха)
  - Transient Loss (потеря транзиентов)
  - Spectral Difference per Second (посекундно)
  - Suspicious Seconds Map (карта проблемных секунд)

v4.1: умная фильтрация — разделяет реальные артефакты и музыкальные события.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class SuperAnalysisError(Exception):
    """Ошибка супер-анализа."""


# ============================================================
#  Базовые утилиты
# ============================================================
def _to_db(x: float) -> float:
    if x <= 1e-10:
        return -200.0
    return 20.0 * math.log10(x)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _rms_envelope(mono: np.ndarray, sr: int, win_sec: float = 0.1) -> np.ndarray:
    win = max(1, int(win_sec * sr))
    n = len(mono) // win
    if n == 0:
        return np.array([])
    frames = mono[:n * win].reshape(n, win)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def _spectral_flatness(frame: np.ndarray) -> float:
    if frame.size < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(frame)) + 1e-12
    geo = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)
    if arith <= 0:
        return 0.0
    return float(geo / arith)


def _spectral_centroid(frame: np.ndarray, sr: int) -> float:
    if frame.size < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(frame)) + 1e-12
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sr)
    return float(np.sum(freqs * spec) / np.sum(spec))


def _crest_factor(x: np.ndarray) -> float:
    rms = _rms(x)
    if rms <= 1e-10:
        return 0.0
    peak = float(np.max(np.abs(x)))
    return peak / rms


def _approx_lufs(rms_db: float) -> float:
    return rms_db - 3.0


def _correlate_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _band_energy_db(frame: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    if frame.size < 8:
        return -200.0
    spec = np.abs(np.fft.rfft(frame)) ** 2
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sr)
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return -200.0
    energy = float(np.sum(spec[mask]))
    return _to_db(math.sqrt(energy))


# ============================================================
#  Bleed и VPI
# ============================================================
def _compute_bleed_score(v_env, i_env, v_activity_threshold_ratio=0.3,
                         excess_threshold_db=3.0) -> dict:
    n = min(len(v_env), len(i_env))
    if n < 5:
        return {"bleed_score": 0.0, "bleed_pct": 0.0, "excess_zones": [],
                "baseline_db": 0.0, "threshold_db": 0.0, "zones_count": 0}
    v_env = v_env[:n]
    i_env = i_env[:n]
    i_env_db = 20 * np.log10(np.maximum(i_env, 1e-10))
    baseline_db = float(np.median(i_env_db))
    threshold_db = baseline_db + excess_threshold_db
    v_norm = v_env / (np.max(v_env) + 1e-12)
    v_active = v_norm > v_activity_threshold_ratio
    excess_mask = v_active & (i_env_db > threshold_db)
    bleed_score = float(np.mean(excess_mask)) if n > 0 else 0.0
    zones = []
    in_zone = False
    start = 0
    for k in range(n):
        if excess_mask[k] and not in_zone:
            in_zone = True
            start = k
        elif not excess_mask[k] and in_zone:
            in_zone = False
            zones.append({
                "start_sec": float(start),
                "end_sec": float(k),
                "duration_sec": float(k - start),
                "peak_excess_db": float(np.max(i_env_db[start:k] - baseline_db)),
            })
    if in_zone:
        zones.append({
            "start_sec": float(start),
            "end_sec": float(n),
            "duration_sec": float(n - start),
            "peak_excess_db": float(np.max(i_env_db[start:] - baseline_db)),
        })
    return {
        "bleed_score": bleed_score,
        "bleed_pct": round(bleed_score * 100.0, 2),
        "baseline_db": round(baseline_db, 2),
        "threshold_db": round(threshold_db, 2),
        "excess_zones": zones,
        "zones_count": len(zones),
    }


def _compute_vpi(v_mono, i_mono, sr, frame_sec=0.1) -> dict:
    frame_len = max(512, int(frame_sec * sr))
    hop = frame_len // 2
    n = min(len(v_mono), len(i_mono))
    n_frames = max(0, (n - frame_len) // hop)
    if n_frames < 5:
        return {"vpi_db": 0.0, "voice_band_lo": 1000, "voice_band_hi": 4000}
    v_band = np.zeros(n_frames)
    i_band = np.zeros(n_frames)
    v_rms_frames = np.zeros(n_frames)
    for k in range(n_frames):
        s = k * hop
        vf = v_mono[s:s + frame_len]
        inf = i_mono[s:s + frame_len]
        v_band[k] = _band_energy_db(vf, sr, 1000.0, 4000.0)
        i_band[k] = _band_energy_db(inf, sr, 1000.0, 4000.0)
        v_rms_frames[k] = _rms(vf)
    v_rms_db = 20 * np.log10(np.maximum(v_rms_frames, 1e-10))
    v_median = float(np.median(v_rms_db))
    v_active = v_rms_db > (v_median + 6.0)
    v_quiet = v_rms_db < (v_median - 6.0)
    if np.sum(v_active) < 3 or np.sum(v_quiet) < 3:
        return {"vpi_db": 0.0, "voice_band_lo": 1000, "voice_band_hi": 4000}
    i_active_mean = float(np.mean(i_band[v_active]))
    i_quiet_mean = float(np.mean(i_band[v_quiet]))
    vpi_db = i_active_mean - i_quiet_mean
    return {
        "vpi_db": round(vpi_db, 2),
        "voice_band_lo": 1000,
        "voice_band_hi": 4000,
        "instrumental_active_band_db": round(i_active_mean, 2),
        "instrumental_quiet_band_db": round(i_quiet_mean, 2),
        "active_frames": int(np.sum(v_active)),
        "quiet_frames": int(np.sum(v_quiet)),
    }


# ============================================================
#  НОВЫЕ МЕТРИКИ v4
# ============================================================
def _phase_smearing(v_mono, i_mono, sr, frame_sec=0.05) -> dict:
    frame_len = max(512, int(frame_sec * sr))
    hop = frame_len // 2
    n = min(len(v_mono), len(i_mono))
    n_frames = max(0, (n - frame_len) // hop)
    if n_frames < 5:
        return {"crest_active": 0.0, "crest_quiet": 0.0, "phase_smearing_db": 0.0}
    v_rms = np.zeros(n_frames)
    i_crest = np.zeros(n_frames)
    for k in range(n_frames):
        s = k * hop
        vf = v_mono[s:s + frame_len]
        inf = i_mono[s:s + frame_len]
        v_rms[k] = _rms(vf)
        i_crest[k] = _crest_factor(inf)
    v_rms_db = 20 * np.log10(np.maximum(v_rms, 1e-10))
    v_median = float(np.median(v_rms_db))
    v_active = v_rms_db > (v_median + 6.0)
    v_quiet = v_rms_db < (v_median - 6.0)
    if np.sum(v_active) < 3 or np.sum(v_quiet) < 3:
        return {"crest_active": 0.0, "crest_quiet": 0.0, "phase_smearing_db": 0.0}
    crest_active = float(np.mean(i_crest[v_active]))
    crest_quiet = float(np.mean(i_crest[v_quiet]))
    phase_smearing_db = _to_db(crest_active) - _to_db(crest_quiet)
    return {
        "crest_active": round(crest_active, 3),
        "crest_quiet": round(crest_quiet, 3),
        "phase_smearing_db": round(phase_smearing_db, 2),
    }


def _spectral_tilt(v_mono, i_mono, sr, frame_sec=0.05) -> dict:
    frame_len = max(512, int(frame_sec * sr))
    hop = frame_len // 2
    n = min(len(v_mono), len(i_mono))
    n_frames = max(0, (n - frame_len) // hop)
    if n_frames < 5:
        return {"centroid_active": 0.0, "centroid_quiet": 0.0,
                "spectral_tilt_pct": 0.0, "high_freq_delta_db": 0.0}
    v_rms = np.zeros(n_frames)
    i_centroid = np.zeros(n_frames)
    i_high = np.zeros(n_frames)
    i_low = np.zeros(n_frames)
    for k in range(n_frames):
        s = k * hop
        vf = v_mono[s:s + frame_len]
        inf = i_mono[s:s + frame_len]
        v_rms[k] = _rms(vf)
        i_centroid[k] = _spectral_centroid(inf, sr)
        i_high[k] = _band_energy_db(inf, sr, 4000.0, 16000.0)
        i_low[k] = _band_energy_db(inf, sr, 500.0, 4000.0)
    v_rms_db = 20 * np.log10(np.maximum(v_rms, 1e-10))
    v_median = float(np.median(v_rms_db))
    v_active = v_rms_db > (v_median + 6.0)
    v_quiet = v_rms_db < (v_median - 6.0)
    if np.sum(v_active) < 3 or np.sum(v_quiet) < 3:
        return {"centroid_active": 0.0, "centroid_quiet": 0.0,
                "spectral_tilt_pct": 0.0, "high_freq_delta_db": 0.0}
    centroid_active = float(np.mean(i_centroid[v_active]))
    centroid_quiet = float(np.mean(i_centroid[v_quiet]))
    if centroid_quiet > 0:
        spectral_tilt_pct = (centroid_active - centroid_quiet) / centroid_quiet * 100.0
    else:
        spectral_tilt_pct = 0.0
    high_active = float(np.mean(i_high[v_active]))
    high_quiet = float(np.mean(i_high[v_quiet]))
    high_freq_delta_db = high_active - high_quiet
    return {
        "centroid_active": round(centroid_active, 1),
        "centroid_quiet": round(centroid_quiet, 1),
        "spectral_tilt_pct": round(spectral_tilt_pct, 2),
        "high_freq_delta_db": round(high_freq_delta_db, 2),
    }


def _transient_loss(v_mono, i_mono, sr, frame_sec=0.02) -> dict:
    frame_len = max(256, int(frame_sec * sr))
    hop = frame_len // 2
    n = min(len(v_mono), len(i_mono))
    n_frames = max(0, (n - frame_len) // hop)
    if n_frames < 5:
        return {"flux_active": 0.0, "flux_quiet": 0.0, "transient_loss_pct": 0.0}
    v_rms = np.zeros(n_frames)
    i_flux = np.zeros(n_frames)
    prev_spec = None
    for k in range(n_frames):
        s = k * hop
        vf = v_mono[s:s + frame_len]
        inf = i_mono[s:s + frame_len]
        v_rms[k] = _rms(vf)
        spec = np.abs(np.fft.rfft(inf * np.hanning(len(inf))))
        if prev_spec is not None:
            diff = np.diff(spec)
            i_flux[k] = float(np.sum(np.maximum(diff, 0)))
        prev_spec = spec
    v_rms_db = 20 * np.log10(np.maximum(v_rms, 1e-10))
    v_median = float(np.median(v_rms_db))
    v_active = v_rms_db > (v_median + 6.0)
    v_quiet = v_rms_db < (v_median - 6.0)
    if np.sum(v_active) < 3 or np.sum(v_quiet) < 3:
        return {"flux_active": 0.0, "flux_quiet": 0.0, "transient_loss_pct": 0.0}
    flux_active = float(np.mean(i_flux[v_active]))
    flux_quiet = float(np.mean(i_flux[v_quiet]))
    if flux_quiet > 0:
        transient_loss_pct = (flux_active - flux_quiet) / flux_quiet * 100.0
    else:
        transient_loss_pct = 0.0
    return {
        "flux_active": round(flux_active, 4),
        "flux_quiet": round(flux_quiet, 4),
        "transient_loss_pct": round(transient_loss_pct, 2),
    }


def _spectral_difference_per_second(v_mono, i_mono, sr) -> list[dict]:
    n = min(len(v_mono), len(i_mono))
    v_mono = v_mono[:n]
    i_mono = i_mono[:n]
    frame_len = int(0.05 * sr)
    hop = frame_len // 2
    n_frames = max(0, (n - frame_len) // hop)
    v_rms = np.zeros(n_frames)
    i_band_1_4k = np.zeros(n_frames)
    i_centroid = np.zeros(n_frames)
    i_crest = np.zeros(n_frames)
    for k in range(n_frames):
        s = k * hop
        vf = v_mono[s:s + frame_len]
        inf = i_mono[s:s + frame_len]
        v_rms[k] = _rms(vf)
        i_band_1_4k[k] = _band_energy_db(inf, sr, 1000.0, 4000.0)
        i_centroid[k] = _spectral_centroid(inf, sr)
        i_crest[k] = _crest_factor(inf)
    v_rms_db = 20 * np.log10(np.maximum(v_rms, 1e-10))
    v_median = float(np.median(v_rms_db))
    v_active = v_rms_db > (v_median + 6.0)
    v_quiet = v_rms_db < (v_median - 6.0)
    if np.sum(v_quiet) >= 3:
        band_ref = float(np.mean(i_band_1_4k[v_quiet]))
        cent_ref = float(np.mean(i_centroid[v_quiet]))
        crest_ref = float(np.mean(i_crest[v_quiet]))
    else:
        band_ref = float(np.median(i_band_1_4k))
        cent_ref = float(np.median(i_centroid))
        crest_ref = float(np.median(i_crest))
    sec_frames = int(sr / hop)
    n_sec = n_frames // sec_frames
    result = []
    for sec in range(n_sec):
        s = sec * sec_frames
        e = s + sec_frames
        va = v_active[s:e]
        if not np.any(va):
            continue
        band_sec = float(np.mean(i_band_1_4k[s:e][va]))
        cent_sec = float(np.mean(i_centroid[s:e][va]))
        crest_sec = float(np.mean(i_crest[s:e][va]))
        result.append({
            "second": sec,
            "band_delta_db": round(band_sec - band_ref, 2),
            "centroid_delta_hz": round(cent_sec - cent_ref, 1),
            "crest_delta_pct": round((crest_sec - crest_ref) / crest_ref * 100.0, 2) if crest_ref > 0 else 0.0,
            "v_active_pct": round(float(np.mean(va)) * 100.0, 1),
        })
    return result


# ============================================================
#  Основной анализ
# ============================================================
def super_analyze_pair(vocals_path, instrumental_path,
                       report_dir=None, progress_cb=None) -> dict:
    vocals_p = Path(vocals_path)
    instr_p = Path(instrumental_path)

    if not vocals_p.exists():
        raise SuperAnalysisError(f"Вокал не найден: {vocals_p}")
    if not instr_p.exists():
        raise SuperAnalysisError(f"Минус не найден: {instr_p}")

    if report_dir is None:
        report_dir = vocals_p.parent / "analysis"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    base = vocals_p.stem
    if base.endswith("_vocals"):
        base = base[:-len("_vocals")]

    if progress_cb:
        progress_cb(0.02, "Чтение аудио")

    try:
        v_data, sr = sf.read(str(vocals_p), dtype="float32", always_2d=True)
        i_data, sr2 = sf.read(str(instr_p), dtype="float32", always_2d=True)
    except Exception as e:
        raise SuperAnalysisError(f"Ошибка чтения аудио: {e}") from e

    if sr != sr2:
        raise SuperAnalysisError(f"Sample rate не совпадает: {sr} vs {sr2}")

    v_mono = v_data.mean(axis=1)
    i_mono = i_data.mean(axis=1)
    n = min(len(v_mono), len(i_mono))
    v_mono = v_mono[:n]
    i_mono = i_mono[:n]
    duration = n / sr

    if progress_cb:
        progress_cb(0.10, "Уровни и динамика")

    v_peak = float(np.max(np.abs(v_mono))) if n else 0.0
    i_peak = float(np.max(np.abs(i_mono))) if n else 0.0
    v_rms = _rms(v_mono)
    i_rms = _rms(i_mono)
    v_rms_db = _to_db(v_rms)
    i_rms_db = _to_db(i_rms)
    v_peak_db = _to_db(v_peak)
    i_peak_db = _to_db(i_peak)
    v_crest = _crest_factor(v_mono)
    i_crest = _crest_factor(i_mono)
    v_lufs = _approx_lufs(v_rms_db)
    i_lufs = _approx_lufs(i_rms_db)

    if progress_cb:
        progress_cb(0.20, "Огибающие")

    seg_sec = 1.0
    v_env = _rms_envelope(v_mono, sr, seg_sec)
    i_env = _rms_envelope(i_mono, sr, seg_sec)
    n_seg = min(len(v_env), len(i_env))
    v_env = v_env[:n_seg]
    i_env = i_env[:n_seg]
    env_corr = _correlate_1d(v_env, i_env)

    if progress_cb:
        progress_cb(0.28, "Bleed-анализ")
    bleed_info = _compute_bleed_score(v_env, i_env)

    if progress_cb:
        progress_cb(0.36, "VPI-анализ")
    vpi_info = _compute_vpi(v_mono, i_mono, sr)

    if progress_cb:
        progress_cb(0.45, "Phase Smearing")
    phase_info = _phase_smearing(v_mono, i_mono, sr)

    if progress_cb:
        progress_cb(0.55, "Spectral Tilt")
    tilt_info = _spectral_tilt(v_mono, i_mono, sr)

    if progress_cb:
        progress_cb(0.65, "Transient Loss")
    transient_info = _transient_loss(v_mono, i_mono, sr)

    if progress_cb:
        progress_cb(0.78, "Посекундный анализ")
    per_sec = _spectral_difference_per_second(v_mono, i_mono, sr)

    # ---------- УМНАЯ ФИЛЬТРАЦИЯ ----------
    suspicious_seconds = []
    n_per_sec = len(per_sec)

    for i, item in enumerate(per_sec):
        sec = item["second"]
        v_act = item["v_active_pct"]

        prev_band = per_sec[i - 1]["band_delta_db"] if i > 0 else 0.0
        next_band = per_sec[i + 1]["band_delta_db"] if i + 1 < n_per_sec else 0.0
        is_isolated = abs(prev_band) < 3.0 and abs(next_band) < 3.0

        flags = []

        # BLEED — только если вокал АКТИВЕН и это изолированная аномалия
        if item["band_delta_db"] > 3.0 and v_act >= 30.0 and is_isolated:
            flags.append("bleed_1_4k")

        # DARKENING — только если вокал активен
        if item["centroid_delta_hz"] < -200.0 and v_act >= 30.0:
            flags.append("darkening")

        # SMEARING — только если вокал активен
        if item["crest_delta_pct"] < -15.0 and v_act >= 30.0:
            flags.append("smearing")

        # МУЗЫКАЛЬНОЕ СОБЫТИЕ — аномалия есть, но вокал НЕ активен
        if not flags and item["band_delta_db"] > 5.0 and v_act < 15.0:
            flags.append("instrumental_event")

        if flags:
            suspicious_seconds.append({
                "second": sec,
                "flags": flags,
                "band_delta_db": item["band_delta_db"],
                "centroid_delta_hz": item["centroid_delta_hz"],
                "crest_delta_pct": item["crest_delta_pct"],
                "v_active_pct": v_act,
            })

    # Разделяем на реальные артефакты и музыкальные события
    real_artifacts = [s for s in suspicious_seconds if "instrumental_event" not in s["flags"]]
    events = [s for s in suspicious_seconds if "instrumental_event" in s["flags"]]

    if progress_cb:
        progress_cb(0.88, "Графики")

    wave_png = report_dir / f"{base}_super_pair_waveforms.png"
    envelope_png = report_dir / f"{base}_super_pair_envelopes.png"
    suspicious_png = report_dir / f"{base}_super_pair_suspicious.png"

    _save_waveforms(v_mono, i_mono, sr, wave_png)
    _save_envelopes(v_env, i_env, seg_sec, envelope_png)
    _save_suspicious(i_mono, sr, suspicious_seconds, suspicious_png)

    if progress_cb:
        progress_cb(0.95, "Вердикт")

    verdict = _make_verdict_v4(
        v_rms_db, i_rms_db, env_corr, bleed_info, vpi_info,
        phase_info, tilt_info, transient_info,
        real_artifacts, events,
    )

    report = {
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "vocals": str(vocals_p.resolve()),
        "instrumental": str(instr_p.resolve()),
        "duration_sec": round(duration, 2),
        "sample_rate": int(sr),

        "level_1_overall": {
            "vocals_rms_db": round(v_rms_db, 2),
            "instrumental_rms_db": round(i_rms_db, 2),
            "vocals_peak_db": round(v_peak_db, 2),
            "instrumental_peak_db": round(i_peak_db, 2),
            "vocals_lufs_approx": round(v_lufs, 2),
            "instrumental_lufs_approx": round(i_lufs, 2),
            "vocals_crest": round(v_crest, 2),
            "instrumental_crest": round(i_crest, 2),
        },

        "level_2_envelopes": {
            "seg_sec": seg_sec,
            "env_correlation": round(env_corr, 4),
        },

        "level_5_bleed": bleed_info,
        "level_6_vpi": vpi_info,
        "level_8_phase_smearing": phase_info,
        "level_9_spectral_tilt": tilt_info,
        "level_10_transient_loss": transient_info,
        "level_11_per_second": per_sec,

        "level_12_real_artifacts": {
            "count": len(real_artifacts),
            "items": real_artifacts[:50],
        },
        "level_13_instrumental_events": {
            "count": len(events),
            "items": events[:50],
        },

        "verdict": verdict,

        "artifacts": {
            "waveforms_png": str(wave_png),
            "envelopes_png": str(envelope_png),
            "suspicious_png": str(suspicious_png),
        },
    }

    json_path = report_dir / f"{base}_super_pair_analysis.json"
    txt_path = report_dir / f"{base}_super_pair_analysis.txt"

    json_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(_render_text_v4(report), encoding="utf-8")

    if progress_cb:
        progress_cb(1.0, "Готово")

    logger.info(
        "Super-анализ v4.1 завершён",
        extra={
            "ctx_real_artifacts": len(real_artifacts),
            "ctx_events": len(events),
            "ctx_phase_smearing": phase_info.get("phase_smearing_db"),
            "ctx_tilt_pct": tilt_info.get("spectral_tilt_pct"),
        },
    )
    return report


# ============================================================
#  Вердикт
# ============================================================
def _make_verdict_v4(v_rms_db, i_rms_db, env_corr, bleed, vpi, phase, tilt,
                     transient, real_artifacts, events) -> dict:
    kid = []
    pro = []

    # Bleed
    if bleed["bleed_pct"] > 20:
        kid.append(f"⚠️ Вокал слышен в минусе {bleed['bleed_pct']:.0f}% времени.")
    elif bleed["bleed_pct"] > 5:
        kid.append(f"🤔 Немного вокала в минусе — {bleed['bleed_pct']:.0f}%.")
    else:
        kid.append("✅ Вокал в минусе не слышен.")

    # VPI
    if vpi["vpi_db"] > 3.0:
        kid.append(f"⚠️ Голосовой отпечаток заметен (VPI {vpi['vpi_db']:.1f} dB).")
    elif vpi["vpi_db"] > 1.5:
        kid.append(f"🤔 Небольшой отпечаток голоса (VPI {vpi['vpi_db']:.1f} dB).")
    else:
        kid.append(f"✅ Отпечатка нет (VPI {vpi['vpi_db']:.1f} dB).")

    # Phase smearing
    if phase["phase_smearing_db"] < -2.0:
        kid.append(f"⚠️ Атака смазана ({phase['phase_smearing_db']:.1f} dB).")
        pro.append(f"🔴 PHASE SMEARING: {phase['phase_smearing_db']} dB "
                   f"(crest {phase['crest_active']} vs {phase['crest_quiet']})")
    elif phase["phase_smearing_db"] < -0.5:
        pro.append(f"🟡 PHASE SMEARING: {phase['phase_smearing_db']} dB")
    else:
        pro.append(f"🟢 PHASE SMEARING: OK ({phase['phase_smearing_db']} dB)")

    # Spectral tilt
    if tilt["spectral_tilt_pct"] < -5.0:
        kid.append(f"⚠️ Минус темнеет ({tilt['spectral_tilt_pct']:.1f}%).")
        pro.append(f"🔴 SPECTRAL TILT: {tilt['spectral_tilt_pct']}%")
    elif tilt["spectral_tilt_pct"] < -2.0:
        pro.append(f"🟡 SPECTRAL TILT: {tilt['spectral_tilt_pct']}%")
    else:
        pro.append(f"🟢 SPECTRAL TILT: OK ({tilt['spectral_tilt_pct']}%)")

    # Transient loss
    if transient["transient_loss_pct"] < -15.0:
        kid.append(f"⚠️ Атака теряется ({transient['transient_loss_pct']:.1f}%).")
        pro.append(f"🔴 TRANSIENT LOSS: {transient['transient_loss_pct']}%")
    elif transient["transient_loss_pct"] < -5.0:
        pro.append(f"🟡 TRANSIENT LOSS: {transient['transient_loss_pct']}%")
    else:
        pro.append(f"🟢 TRANSIENT LOSS: OK ({transient['transient_loss_pct']}%)")

    # Real artifacts
    if real_artifacts:
        kid.append(f"📍 Подозрительных секунд: {len(real_artifacts)}.")
        pro.append("")
        pro.append(f"📍 РЕАЛЬНЫЕ АРТЕФАКТЫ ({len(real_artifacts)} шт):")
        for s in real_artifacts[:20]:
            pro.append(f"   {s['second']:>4} сек: "
                       f"flags={s['flags']}, "
                       f"band={s['band_delta_db']:+.1f} dB, "
                       f"centroid={s['centroid_delta_hz']:+.0f} Hz, "
                       f"crest={s['crest_delta_pct']:+.1f}%, "
                       f"v_active={s['v_active_pct']:.0f}%")
    else:
        kid.append("✅ Реальных артефактов нет.")
        pro.append("🟢 РЕАЛЬНЫХ АРТЕФАКТОВ: 0")

    # Musical events (не артефакты)
    if events:
        pro.append("")
        pro.append(f"🎵 МУЗЫКАЛЬНЫЕ СОБЫТИЯ (не артефакты, {len(events)} шт):")
        pro.append("   (смена секции / соло / крещендо — это норма)")
        for s in events[:15]:
            pro.append(f"   {s['second']:>4} сек: "
                       f"band={s['band_delta_db']:+.1f} dB, "
                       f"v_active={s['v_active_pct']:.0f}%")

    pro.append("")
    pro.append("📌 РЕКОМЕНДАЦИИ:")

    if phase["phase_smearing_db"] < -2.0 or transient["transient_loss_pct"] < -10.0:
        pro.append("  1. 🔴 Проблема с транзиентами — попробуй:")
        pro.append("     - BS-Roformer: Gabox 🎸 bleedless v2")
        pro.append("     - BS-Roformer: Becruily 🎤 karaoke")
        pro.append("  2. Или ослабь Notch в separator_bs.py: INSTR_NOTCH_GAIN_DB = -6.0")
        pro.append("     (фактическая глубина ≈ -6 dB вместо ≈ -16 dB).")
        pro.append("     ⚠️ Инверсия параметра: чем БЛИЖЕ значение к 0, тем ГЛУБЖЕ notch")
        pro.append("     (-1.5 → ≈-16 dB, -6.0 → ≈-6 dB, -10.0 → ≈-3.3 dB).")

    if tilt["spectral_tilt_pct"] < -5.0:
        pro.append("  3. 🔴 Минус темнеет — сначала проверь причину:")
        pro.append("     - в логе строка 'Notch: ... фактическая глубина' ≈ -16 dB?")
        pro.append("       Тогда виноват Notch, а не LPF → INSTR_NOTCH_GAIN_DB = -6.0")
        pro.append("     - LPF выставлен ниже 19 кГц? → подними до 19 кГц")
        pro.append("     - или убери LPF совсем, оставь только Notch.")

    if bleed["bleed_pct"] > 20:
        pro.append("  5. 🔴 Высокий bleed — перейди на Becruily 🎤 karaoke.")

    if not real_artifacts and bleed["bleed_pct"] < 5 and vpi["vpi_db"] < 2.0:
        pro.append("  ✅ Минус чистый. Вмешательство не требуется.")

    return {
        "kid": " ".join(kid),
        "pro": "\n".join(pro),
    }


# ============================================================
#  Графики
# ============================================================
def _save_waveforms(v_mono, i_mono, sr, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), dpi=100, sharex=True)
    step = max(1, len(v_mono) // 8000)
    t = np.arange(0, len(v_mono), step) / sr
    axes[0].plot(t, v_mono[::step], linewidth=0.4, color="#e74c3c")
    axes[0].set_ylabel("Vocals")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, i_mono[::step], linewidth=0.4, color="#3498db")
    axes[1].set_ylabel("Instrumental")
    axes[1].set_xlabel("сек")
    axes[1].grid(alpha=0.3)
    axes[0].set_title("Waveforms: Vocals vs Instrumental")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_envelopes(v_env, i_env, seg_sec, out_path):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    t = np.arange(len(v_env)) * seg_sec
    v_db = 20 * np.log10(np.maximum(v_env, 1e-10))
    i_db = 20 * np.log10(np.maximum(i_env, 1e-10))
    ax.plot(t, v_db, linewidth=1.2, color="#e74c3c", label="Vocals")
    ax.plot(t, i_db, linewidth=1.2, color="#3498db", label="Instrumental")
    ax.set_xlabel("сек")
    ax.set_ylabel("RMS, dB")
    ax.set_title("RMS Envelopes")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_suspicious(i_mono, sr, suspicious, out_path):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
    step = max(1, len(i_mono) // 12000)
    t = np.arange(0, len(i_mono), step) / sr
    ax.plot(t, i_mono[::step], linewidth=0.4, color="#3498db", label="Instrumental")

    for s in suspicious:
        sec = s["second"]
        flags = s["flags"]
        if "bleed_1_4k" in flags:
            color = "#e74c3c"
        elif "darkening" in flags:
            color = "#f39c12"
        elif "smearing" in flags:
            color = "#8e44ad"
        elif "instrumental_event" in flags:
            color = "#2ecc71"
        else:
            color = "#95a5a6"
        ax.axvspan(sec, sec + 1, color=color, alpha=0.4)

    ax.set_xlabel("сек")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Suspicious Seconds ({len(suspicious)} шт) — "
                 f"красный=bleed, оранж=darkening, фиолет=smearing, зелёный=событие")
    ax.grid(alpha=0.3)
    if suspicious:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
#  Текстовый отчёт
# ============================================================
def _render_text_v4(r: dict) -> str:
    L = []
    L.append("=" * 72)
    L.append("🔬 СУПЕР-АНАЛИЗ v4.1: ВОКАЛ + МИНУС")
    L.append("=" * 72)
    L.append(f"Время:        {r['analyzed_at']}")
    L.append(f"Длительность: {r['duration_sec']} сек")
    L.append(f"Sample rate:  {r['sample_rate']} Гц")
    L.append("")

    L.append("─" * 72)
    L.append("📊 УРОВЕНЬ 1. ОБЩИЕ УРОВНИ")
    L.append("─" * 72)
    o = r["level_1_overall"]
    L.append(f"  Вокал   RMS: {o['vocals_rms_db']:>7.2f} dBFS   "
             f"Peak: {o['vocals_peak_db']:>7.2f} dBFS")
    L.append(f"  Минус   RMS: {o['instrumental_rms_db']:>7.2f} dBFS   "
             f"Peak: {o['instrumental_peak_db']:>7.2f} dBFS")
    L.append("")

    L.append("─" * 72)
    L.append("🩸 УРОВЕНЬ 5. BLEED")
    L.append("─" * 72)
    b = r["level_5_bleed"]
    L.append(f"  Bleed:     {b['bleed_pct']}%")
    L.append("")

    L.append("─" * 72)
    L.append("🎤 УРОВЕНЬ 6. VPI")
    L.append("─" * 72)
    v = r["level_6_vpi"]
    L.append(f"  VPI: {v['vpi_db']} dB")
    L.append("")

    L.append("─" * 72)
    L.append("🌀 УРОВЕНЬ 8. PHASE SMEARING")
    L.append("─" * 72)
    p = r["level_8_phase_smearing"]
    L.append(f"  Crest в вокале:  {p['crest_active']}")
    L.append(f"  Crest в тишине:  {p['crest_quiet']}")
    L.append(f"  Phase Smearing:  {p['phase_smearing_db']} dB")
    L.append("")

    L.append("─" * 72)
    L.append("🌑 УРОВЕНЬ 9. SPECTRAL TILT")
    L.append("─" * 72)
    t = r["level_9_spectral_tilt"]
    L.append(f"  Centroid в вокале:  {t['centroid_active']} Гц")
    L.append(f"  Centroid в тишине:  {t['centroid_quiet']} Гц")
    L.append(f"  Tilt:               {t['spectral_tilt_pct']}%")
    L.append("")

    L.append("─" * 72)
    L.append("💥 УРОВЕНЬ 10. TRANSIENT LOSS")
    L.append("─" * 72)
    tr = r["level_10_transient_loss"]
    L.append(f"  Transient Loss: {tr['transient_loss_pct']}%")
    L.append("")

    L.append("─" * 72)
    L.append("📍 УРОВЕНЬ 12. РЕАЛЬНЫЕ АРТЕФАКТЫ")
    L.append("─" * 72)
    ra = r["level_12_real_artifacts"]
    L.append(f"  Найдено: {ra['count']}")
    for item in ra["items"][:30]:
        L.append(f"    Сек {item['second']:>4}: "
                 f"flags={item['flags']}, "
                 f"band={item['band_delta_db']:+.1f} dB, "
                 f"centroid={item['centroid_delta_hz']:+.0f} Hz, "
                 f"crest={item['crest_delta_pct']:+.1f}%, "
                 f"v_active={item['v_active_pct']:.0f}%")
    L.append("")

    L.append("─" * 72)
    L.append("🎵 УРОВЕНЬ 13. МУЗЫКАЛЬНЫЕ СОБЫТИЯ (не артефакты)")
    L.append("─" * 72)
    ev = r["level_13_instrumental_events"]
    L.append(f"  Найдено: {ev['count']}")
    L.append("  (смена секции / соло / крещендо — это норма)")
    for item in ev["items"][:20]:
        L.append(f"    Сек {item['second']:>4}: "
                 f"band={item['band_delta_db']:+.1f} dB, "
                 f"v_active={item['v_active_pct']:.0f}%")
    L.append("")

    L.append("=" * 72)
    L.append("👶 ВЕРДИКТ ПРОСТЫМИ СЛОВАМИ")
    L.append("=" * 72)
    L.append("  " + r["verdict"]["kid"])
    L.append("")

    L.append("=" * 72)
    L.append("👨‍🔬 ВЕРДИКТ ДЛЯ ПРОФЕССИОНАЛА")
    L.append("=" * 72)
    for line in r["verdict"]["pro"].split("\n"):
        L.append("  " + line)
    L.append("")

    L.append("─" * 72)
    L.append("📁 АРТЕФАКТЫ")
    L.append("─" * 72)
    for k, vv in r["artifacts"].items():
        L.append(f"  {k}: {vv}")
    L.append("=" * 72)

    return "\n".join(L)