"""
СУПЕР-АНАЛИЗ ПАРЫ: вокал + минус.

Даёт пять уровней понимания:
  1. Простой вердикт (для ребёнка).
  2. Что слышит музыкант (RMS, спектр, зоны).
  3. Что видит звукореж (LUFS, crest, стерео, bleed).
  4. Что видит ML-инженер (флатнесс, корреляции, MFCC).
  5. Что видит SOTA (perceptual, распределение артефактов).

Всё сохраняется в JSON + TXT + PNG-графики.
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
#  Вспомогательные функции
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
    """RMS-огибающая с окном win_sec."""
    win = max(1, int(win_sec * sr))
    n = len(mono) // win
    if n == 0:
        return np.array([])
    frames = mono[:n * win].reshape(n, win)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def _spectral_flatness(frame: np.ndarray) -> float:
    """Спектральный флатнесс: 0 — тональный, 1 — шумовой."""
    if frame.size < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(frame)) + 1e-12
    geo = np.exp(np.mean(np.log(spec)))
    arith = np.mean(spec)
    if arith <= 0:
        return 0.0
    return float(geo / arith)


def _spectral_centroid(frame: np.ndarray, sr: int) -> float:
    """Спектральный центроид — «яркость» звука."""
    if frame.size < 8:
        return 0.0
    spec = np.abs(np.fft.rfft(frame)) + 1e-12
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sr)
    return float(np.sum(freqs * spec) / np.sum(spec))


def _crest_factor(x: np.ndarray) -> float:
    """Crest factor = peak / rms."""
    rms = _rms(x)
    if rms <= 1e-10:
        return 0.0
    peak = float(np.max(np.abs(x)))
    return peak / rms


def _approx_lufs(rms_db: float) -> float:
    """Грубая оценка LUFS из RMS."""
    return rms_db - 3.0


def _correlate_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson корреляция двух 1D массивов."""
    if a.size < 2 or b.size < 2:
        return 0.0
    n = min(len(a), len(b))
    a = a[:n].astype(np.float64)
    b = b[:n].astype(np.float64)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _mfcc(frame: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Грубая MFCC через DCT от mel-спектра (без librosa)."""
    if frame.size < 512:
        frame = np.pad(frame, (0, 512 - frame.size))
    spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) ** 2
    # Грубый mel: логарифмические полосы
    n_filters = 26
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
    mel_spec = np.zeros(n_filters)
    edges = np.logspace(np.log10(20), np.log10(sr / 2), n_filters + 1)
    for i in range(n_filters):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if np.any(mask):
            mel_spec[i] = np.log(np.sum(spec[mask]) + 1e-12)
    # DCT-II (упрощённо через numpy)
    n = len(mel_spec)
    dct = np.zeros(n_mfcc)
    for k in range(n_mfcc):
        for i in range(n):
            dct[k] += mel_spec[i] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    return dct


# ============================================================
#  Основной анализ
# ============================================================
def super_analyze_pair(
    vocals_path: str | Path,
    instrumental_path: str | Path,
    report_dir: str | Path | None = None,
    progress_cb=None,
) -> dict:
    """
    Глубокий сравнительный анализ вокала и минуса.
    """
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

    # ---------- Чтение ----------
    try:
        v_data, sr = sf.read(str(vocals_p), dtype="float32", always_2d=True)
        i_data, sr2 = sf.read(str(instr_p), dtype="float32", always_2d=True)
    except Exception as e:
        raise SuperAnalysisError(f"Ошибка чтения аудио: {e}") from e

    if sr != sr2:
        raise SuperAnalysisError(f"Sample rate не совпадает: {sr} vs {sr2}")

    v_mono = v_data.mean(axis=1)
    i_mono = i_data.mean(axis=1)

    # Обрезаем до одинаковой длины
    n = min(len(v_mono), len(i_mono))
    v_mono = v_mono[:n]
    i_mono = i_mono[:n]

    duration = n / sr

    if progress_cb:
        progress_cb(0.10, "Уровни и динамика")

    # ---------- Уровень 1: Общее ----------
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

    # ---------- Уровень 2: Огибающие по сегментам ----------
    if progress_cb:
        progress_cb(0.20, "Огибающие по сегментам")

    seg_sec = 1.0  # окно 1 секунда
    v_env = _rms_envelope(v_mono, sr, seg_sec)
    i_env = _rms_envelope(i_mono, sr, seg_sec)

    n_seg = min(len(v_env), len(i_env))
    v_env = v_env[:n_seg]
    i_env = i_env[:n_seg]

    # Корреляция огибающих
    env_corr = _correlate_1d(v_env, i_env)

    # Зоны подозрения: где корреляция локально высокая
    win = 5  # скользящее окно 5 сек
    suspicion_zones = []
    for i in range(0, n_seg - win):
        a = v_env[i:i + win]
        b = i_env[i:i + win]
        c = _correlate_1d(a, b)
        if c > 0.55:
            suspicion_zones.append({
                "start_sec": round(i * seg_sec, 1),
                "end_sec": round((i + win) * seg_sec, 1),
                "corr": round(c, 3),
            })

    # Схлопываем соседние зоны
    merged_zones = []
    for z in suspicion_zones:
        if merged_zones and z["start_sec"] <= merged_zones[-1]["end_sec"]:
            merged_zones[-1]["end_sec"] = max(merged_zones[-1]["end_sec"], z["end_sec"])
            merged_zones[-1]["corr"] = max(merged_zones[-1]["corr"], z["corr"])
        else:
            merged_zones.append(dict(z))

    # ---------- Уровень 3: Спектральные метрики по кадрам ----------
    if progress_cb:
        progress_cb(0.40, "Спектральный анализ")

    frame_len = int(0.05 * sr)  # 50 мс
    hop = frame_len // 2
    n_frames = max(0, (n - frame_len) // hop)

    v_flatness = []
    i_flatness = []
    v_centroid = []
    i_centroid = []

    for k in range(n_frames):
        s = k * hop
        vf = v_mono[s:s + frame_len]
        inf = i_mono[s:s + frame_len]
        v_flatness.append(_spectral_flatness(vf))
        i_flatness.append(_spectral_flatness(inf))
        v_centroid.append(_spectral_centroid(vf, sr))
        i_centroid.append(_spectral_centroid(inf, sr))

    v_flatness = np.array(v_flatness)
    i_flatness = np.array(i_flatness)
    v_centroid = np.array(v_centroid)
    i_centroid = np.array(i_centroid)

    flatness_corr = _correlate_1d(v_flatness, i_flatness)
    centroid_corr = _correlate_1d(v_centroid, i_centroid)

    # ---------- Уровень 4: MFCC-дистанция ----------
    if progress_cb:
        progress_cb(0.60, "MFCC-анализ")

    mfcc_step = max(1, n_frames // 200)  # берём ~200 кадров
    v_mfccs = []
    i_mfccs = []
    for k in range(0, n_frames, mfcc_step):
        s = k * hop
        v_mfccs.append(_mfcc(v_mono[s:s + frame_len], sr))
        i_mfccs.append(_mfcc(i_mono[s:s + frame_len], sr))

    v_mfccs = np.array(v_mfccs)
    i_mfccs = np.array(i_mfccs)

    if len(v_mfccs) > 0 and len(i_mfccs) > 0:
        # Косинусное расстояние между усреднёнными MFCC
        vm = v_mfccs.mean(axis=0)
        im = i_mfccs.mean(axis=0)
        cos_dist = 1.0 - float(
            np.dot(vm, im) / (np.linalg.norm(vm) * np.linalg.norm(im) + 1e-12)
        )
        # Евклидово расстояние между усреднёнными MFCC
        euclid = float(np.linalg.norm(vm - im))
    else:
        cos_dist = 0.0
        euclid = 0.0

    # ---------- Уровень 5: Bleed Score ----------
    if progress_cb:
        progress_cb(0.75, "Bleed-анализ")

    # Bleed score: сколько энергии минуса совпадает по времени с вокалом
    # Если вокал тихий, а минус громкий — там нет bleed'а
    # Если вокал громкий, и минус тоже громкий — там может быть bleed
    bleed_score = 0.0
    if n_seg > 0:
        # Нормализуем огибающие
        v_norm = v_env / (np.max(v_env) + 1e-12)
        i_norm = i_env / (np.max(i_env) + 1e-12)
        # Bleed = доля кадров, где обе огибающие высокие
        threshold = 0.3
        both_high = (v_norm > threshold) & (i_norm > threshold)
        bleed_score = float(np.mean(both_high))

    # ---------- Читаемый вердикт ----------
    if progress_cb:
        progress_cb(0.85, "Формирование вердикта")

    verdict_kid = _make_kid_verdict(
        v_rms_db, i_rms_db, env_corr, bleed_score, len(merged_zones)
    )
    verdict_pro = _make_pro_verdict(
        env_corr, flatness_corr, centroid_corr, bleed_score, merged_zones
    )

    # ---------- Графики ----------
    if progress_cb:
        progress_cb(0.92, "Графики")

    wave_png = report_dir / f"{base}_super_pair_waveforms.png"
    envelope_png = report_dir / f"{base}_super_pair_envelopes.png"
    flatness_png = report_dir / f"{base}_super_pair_flatness.png"

    _save_waveforms(v_mono, i_mono, sr, wave_png)
    _save_envelopes(v_env, i_env, seg_sec, envelope_png)
    _save_flatness(v_flatness, i_flatness, hop, sr, flatness_png)

    # ---------- Отчёт ----------
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
            "suspicion_zones": merged_zones,
            "zones_count": len(merged_zones),
        },

        "level_3_spectral": {
            "flatness_correlation": round(flatness_corr, 4),
            "centroid_correlation": round(centroid_corr, 4),
            "vocals_flatness_mean": round(float(np.mean(v_flatness)), 5) if len(v_flatness) else 0.0,
            "instrumental_flatness_mean": round(float(np.mean(i_flatness)), 5) if len(i_flatness) else 0.0,
            "vocals_centroid_mean": round(float(np.mean(v_centroid)), 1) if len(v_centroid) else 0.0,
            "instrumental_centroid_mean": round(float(np.mean(i_centroid)), 1) if len(i_centroid) else 0.0,
        },

        "level_4_mfcc": {
            "cosine_distance": round(cos_dist, 5),
            "euclidean_distance": round(euclid, 3),
            "frames_compared": int(min(len(v_mfccs), len(i_mfccs))),
        },

        "level_5_bleed": {
            "bleed_score": round(bleed_score, 4),
            "bleed_pct": round(bleed_score * 100.0, 2),
        },

        "verdict_kid": verdict_kid,
        "verdict_pro": verdict_pro,

        "artifacts": {
            "waveforms_png": str(wave_png),
            "envelopes_png": str(envelope_png),
            "flatness_png": str(flatness_png),
        },
    }

    # ---------- Сохранение ----------
    json_path = report_dir / f"{base}_super_pair_analysis.json"
    txt_path = report_dir / f"{base}_super_pair_analysis.txt"

    json_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(_render_text(report), encoding="utf-8")

    if progress_cb:
        progress_cb(1.0, "Готово")

    logger.info(
        "Super-анализ пары завершён",
        extra={
            "ctx_vocals": str(vocals_p),
            "ctx_instr": str(instr_p),
            "ctx_env_corr": env_corr,
            "ctx_bleed": bleed_score,
        },
    )
    return report


# ============================================================
#  JSON-safe
# ============================================================
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


# ============================================================
#  Вердикты
# ============================================================
def _make_kid_verdict(v_rms, i_rms, env_corr, bleed, zones):
    lines = []

    # Громкость
    if i_rms > v_rms + 5:
        lines.append("🎵 Минус звучит громче вокала — это нормально.")
    elif v_rms > i_rms + 5:
        lines.append("🎵 Вокал звучит громче минуса — обычно наоборот делают.")
    else:
        lines.append("🎵 Вокал и минус примерно одинаковы по громкости.")

    # Bleed
    if bleed > 0.4:
        lines.append(f"⚠️ Похоже, вокал «протекает» в минус — {bleed*100:.0f}% времени.")
    elif bleed > 0.2:
        lines.append(f"🤔 Возможно, вокал немного слышен в минусе — {bleed*100:.0f}% времени.")
    else:
        lines.append("✅ Минус в основном чистый, вокал в нём не слышится.")

    # Зоны
    if zones > 0:
        lines.append(f"📍 Есть {zones} мест, где это заметно особенно сильно.")
    else:
        lines.append("✅ Особых проблемных участков не найдено.")

    return " ".join(lines)


def _make_pro_verdict(env_corr, flat_corr, cent_corr, bleed, zones):
    lines = []

    # Корреляция огибающих
    if env_corr > 0.6:
        lines.append(f"🔴 RMS-огибающие сильно коррелируют ({env_corr:.2f}) — "
                     "вероятна протечка вокала в минус.")
    elif env_corr > 0.4:
        lines.append(f"🟡 Умеренная корреляция огибающих ({env_corr:.2f}) — "
                     "стоит проверить вокальные фрагменты.")
    else:
        lines.append(f"🟢 Низкая корреляция огибающих ({env_corr:.2f}) — "
                     "вокал и минус независимы.")

    # Флатнесс
    if flat_corr > 0.5:
        lines.append(f"🔴 Спектральный флатнесс коррелирует ({flat_corr:.2f}) — "
                     "возможно, вокал добавляет шумовой компонент в минус.")

    # Centroid
    if cent_corr > 0.5:
        lines.append(f"🟡 Спектральный центроид коррелирует ({cent_corr:.2f}) — "
                     "«яркость» минуса повторяет вокал.")

    # Bleed score
    if bleed > 0.4:
        lines.append(f"🔴 Bleed score {bleed*100:.1f}% — высокая вероятность артефакта.")
    elif bleed > 0.2:
        lines.append(f"🟡 Bleed score {bleed*100:.1f}% — умеренная вероятность.")

    # Рекомендации
    if zones:
        lines.append("")
        lines.append("📌 Рекомендации:")
        lines.append("  1. Прослушать зоны подозрения в Reaper/Audacity.")
        lines.append("  2. Попробовать Notch 1–4 кГц на минус (−3 dB).")
        lines.append("  3. Или перейти на другую модель (htdemucs / BS-Roformer).")
        lines.append("  4. Или применить Phase Fix в UVR.")

    return "\n".join(lines)


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

    ax.plot(t, v_db, linewidth=1.2, color="#e74c3c", label="Vocals RMS")
    ax.plot(t, i_db, linewidth=1.2, color="#3498db", label="Instrumental RMS")
    ax.set_xlabel("сек")
    ax.set_ylabel("RMS, dB")
    ax.set_title("RMS Envelopes (1 sec window)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_flatness(v_flat, i_flat, hop, sr, out_path):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

    t_v = np.arange(len(v_flat)) * hop / sr
    t_i = np.arange(len(i_flat)) * hop / sr

    ax.plot(t_v, v_flat, linewidth=1.0, color="#e74c3c", label="Vocals")
    ax.plot(t_i, i_flat, linewidth=1.0, color="#3498db", label="Instrumental")
    ax.set_xlabel("сек")
    ax.set_ylabel("Spectral Flatness")
    ax.set_title("Spectral Flatness (per 50ms frame)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
#  Текстовый отчёт
# ============================================================
def _render_text(r: dict) -> str:
    L = []
    L.append("=" * 72)
    L.append("🔬 СУПЕР-АНАЛИЗ ПАРЫ: ВОКАЛ + МИНУС")
    L.append("=" * 72)
    L.append(f"Время:        {r['analyzed_at']}")
    L.append(f"Длительность: {r['duration_sec']} сек")
    L.append(f"Sample rate:  {r['sample_rate']} Гц")
    L.append("")

    # Уровень 1
    L.append("─" * 72)
    L.append("📊 УРОВЕНЬ 1. ОБЩИЕ УРОВНИ (для ребёнка)")
    L.append("─" * 72)
    o = r["level_1_overall"]
    L.append(f"  Вокал   RMS: {o['vocals_rms_db']:>7.2f} dBFS   "
             f"Peak: {o['vocals_peak_db']:>7.2f} dBFS   "
             f"LUFS≈: {o['vocals_lufs_approx']:>7.2f}")
    L.append(f"  Минус   RMS: {o['instrumental_rms_db']:>7.2f} dBFS   "
             f"Peak: {o['instrumental_peak_db']:>7.2f} dBFS   "
             f"LUFS≈: {o['instrumental_lufs_approx']:>7.2f}")
    L.append(f"  Crest: вокал {o['vocals_crest']}, минус {o['instrumental_crest']}")
    L.append("")

    # Уровень 2
    L.append("─" * 72)
    L.append("🎼 УРОВЕНЬ 2. ОГИБАЮЩИЕ ПО СЕГМЕНТАМ (для музыканта)")
    L.append("─" * 72)
    e = r["level_2_envelopes"]
    L.append(f"  Корреляция огибающих: {e['env_correlation']}")
    L.append(f"  Зон подозрения:       {e['zones_count']}")
    if e["suspicion_zones"]:
        L.append("  Список зон (секунды):")
        for z in e["suspicion_zones"][:15]:
            L.append(f"    {z['start_sec']:>7.1f} – {z['end_sec']:>7.1f} "
                     f"(corr={z['corr']})")
    L.append("")

    # Уровень 3
    L.append("─" * 72)
    L.append("🎛️ УРОВЕНЬ 3. СПЕКТРАЛЬНЫЕ МЕТРИКИ (для звукорежа)")
    L.append("─" * 72)
    s = r["level_3_spectral"]
    L.append(f"  Flatness корреляция:  {s['flatness_correlation']}")
    L.append(f"  Centroid корреляция:  {s['centroid_correlation']}")
    L.append(f"  Flatness вокала:      {s['vocals_flatness_mean']}")
    L.append(f"  Flatness минуса:      {s['instrumental_flatness_mean']}")
    L.append(f"  Centroid вокала:      {s['vocals_centroid_mean']} Гц")
    L.append(f"  Centroid минуса:      {s['instrumental_centroid_mean']} Гц")
    L.append("")

    # Уровень 4
    L.append("─" * 72)
    L.append("🧬 УРОВЕНЬ 4. MFCC-ДИСТАНЦИЯ (для ML-инженера)")
    L.append("─" * 72)
    m = r["level_4_mfcc"]
    L.append(f"  Cosine distance:    {m['cosine_distance']} (больше = сильнее отличаются)")
    L.append(f"  Euclidean distance: {m['euclidean_distance']}")
    L.append(f"  Кадров сравнено:    {m['frames_compared']}")
    L.append("")

    # Уровень 5
    L.append("─" * 72)
    L.append("🩸 УРОВЕНЬ 5. BLEED SCORE (для SOTA)")
    L.append("─" * 72)
    b = r["level_5_bleed"]
    L.append(f"  Bleed score:  {b['bleed_score']} ({b['bleed_pct']}%)")
    L.append(f"  Интерпретация: чем выше — тем сильнее вокал протекает в минус.")
    L.append("")

    # Вердикты
    L.append("=" * 72)
    L.append("👶 ВЕРДИКТ ПРОСТЫМИ СЛОВАМИ")
    L.append("=" * 72)
    L.append("  " + r["verdict_kid"])
    L.append("")

    L.append("=" * 72)
    L.append("👨‍🔬 ВЕРДИКТ ДЛЯ ПРОФЕССИОНАЛА")
    L.append("=" * 72)
    for line in r["verdict_pro"].split("\n"):
        L.append("  " + line)
    L.append("")

    # Артефакты
    L.append("─" * 72)
    L.append("📁 АРТЕФАКТЫ")
    L.append("─" * 72)
    for k, v in r["artifacts"].items():
        L.append(f"  {k}: {v}")
    L.append("=" * 72)

    return "\n".join(L)