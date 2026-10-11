"""
Анализ MIDI-файла: диапазон нот, распределение по октавам, плотность,
средняя высота, темп, длительность. Сохраняет отчёт в JSON/TXT.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class MidiAnalysisError(Exception):
    """Ошибка анализа MIDI."""


# ============================================================
#  Универсальный конвертер numpy → Python для JSON
# ============================================================
def _json_safe(obj):
    """
    Рекурсивно приводит numpy-типы к чистым Python-типам,
    чтобы json.dumps мог их сериализовать.
    """
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


def _note_name(pitch: int) -> str:
    """MIDI-номер → имя ноты (C4 = 60)."""
    pitch = int(pitch)
    octave = pitch // 12 - 1
    return f"{NOTE_NAMES[pitch % 12]}{octave}"


def _octave_label(pitch: int) -> int:
    """MIDI-номер → номер октавы (C4 = 60 → октава 4)."""
    return int(pitch) // 12 - 1


# ============================================================
#  Основной анализ
# ============================================================
def analyze_midi(midi_path: str | Path, report_dir: str | Path | None = None) -> dict:
    """
    Анализ MIDI-файла.

    Возвращает dict со всеми метриками. Также пишет:
      - <report_dir>/<name>_midi_analysis.json
      - <report_dir>/<name>_midi_analysis.txt
    """
    p = Path(midi_path)
    if not p.exists():
        raise MidiAnalysisError(f"MIDI не найден: {p}")

    if report_dir is None:
        report_dir = p.parent / "analysis"
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    base = p.stem

    # ---------- Чтение ----------
    try:
        midi = pretty_midi.PrettyMIDI(str(p))
    except Exception as e:
        logger.exception("Не удалось прочитать MIDI")
        raise MidiAnalysisError(f"Ошибка чтения: {e}") from e

    # ---------- Общая инфо ----------
    duration = float(midi.get_end_time())
    try:
        tempo = float(midi.estimate_tempo()) if duration > 0 else 120.0
    except Exception:
        tempo = 120.0
    n_instruments = len(midi.instruments)

    # ---------- Собираем все ноты ----------
    all_notes = []
    instruments_info = []
    for instr in midi.instruments:
        if not instr.notes:
            continue
        pitches = [int(n.pitch) for n in instr.notes]
        velocities = [int(n.velocity) for n in instr.notes]
        durations = [float(n.end) - float(n.start) for n in instr.notes]

        instruments_info.append({
            "name": str(instr.name or "unnamed"),
            "program": int(instr.program),
            "is_drum": bool(instr.is_drum),
            "notes_count": int(len(instr.notes)),
            "pitch_min": int(min(pitches)),
            "pitch_max": int(max(pitches)),
            "pitch_min_name": _note_name(min(pitches)),
            "pitch_max_name": _note_name(max(pitches)),
            "velocity_avg": float(sum(velocities) / len(velocities)),
            "duration_avg_sec": float(sum(durations) / len(durations)),
        })
        all_notes.extend(instr.notes)

    if not all_notes:
        raise MidiAnalysisError("В MIDI нет нот")

    # ---------- Диапазон ----------
    all_pitches = [int(n.pitch) for n in all_notes]
    pitch_min = int(min(all_pitches))
    pitch_max = int(max(all_pitches))
    pitch_avg = float(sum(all_pitches) / len(all_pitches))
    sorted_pitches = sorted(all_pitches)
    pitch_median = int(sorted_pitches[len(sorted_pitches) // 2])

    # ---------- Распределение по октавам ----------
    octave_counter: Counter = Counter()
    for pch in all_pitches:
        octave_counter[_octave_label(pch)] += 1

    octave_dist = {}
    for oct_num in sorted(octave_counter.keys()):
        count = int(octave_counter[oct_num])
        octave_dist[str(oct_num)] = {
            "octave": int(oct_num),
            "count": count,
            "percent": float(round(100.0 * count / len(all_pitches), 2)),
        }

    # ---------- Распределение по нотам ----------
    note_counter: Counter = Counter()
    for pch in all_pitches:
        note_counter[_note_name(pch)] += 1
    top_notes = [
        {"note": str(n), "count": int(c)}
        for n, c in note_counter.most_common(10)
    ]

    # ---------- Плотность ----------
    notes_per_second = float(round(len(all_notes) / duration, 2)) if duration > 0 else 0.0

    # ---------- Velocity ----------
    velocities = [int(n.velocity) for n in all_notes]
    velocity_min = int(min(velocities))
    velocity_max = int(max(velocities))
    velocity_avg = float(round(sum(velocities) / len(velocities), 1))

    # ---------- Отчёт ----------
    report = {
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "file": str(p.resolve()),
        "name": str(p.name),
        "duration_sec": float(round(duration, 2)),
        "tempo_bpm": float(round(tempo, 1)),
        "instruments": instruments_info,
        "summary": {
            "total_notes": int(len(all_notes)),
            "notes_per_second": notes_per_second,
            "pitch_min": pitch_min,
            "pitch_min_name": _note_name(pitch_min),
            "pitch_max": pitch_max,
            "pitch_max_name": _note_name(pitch_max),
            "pitch_avg": float(round(pitch_avg, 2)),
            "pitch_avg_name": _note_name(int(round(pitch_avg))),
            "pitch_median": pitch_median,
            "pitch_median_name": _note_name(pitch_median),
            "pitch_range_semitones": int(pitch_max - pitch_min),
            "velocity_min": velocity_min,
            "velocity_max": velocity_max,
            "velocity_avg": velocity_avg,
        },
        "octave_distribution": octave_dist,
        "top_10_notes": top_notes,
    }

    # ---------- Запись отчётов ----------
    json_path = report_dir / f"{base}_midi_analysis.json"
    txt_path = report_dir / f"{base}_midi_analysis.txt"

    json_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    txt_path.write_text(_render_text_report(report), encoding="utf-8")

    logger.info(
        "MIDI-анализ завершён",
        extra={
            "ctx_file": str(p),
            "ctx_json": str(json_path),
            "ctx_notes": int(len(all_notes)),
        },
    )
    return report


# ============================================================
#  Читаемый текстовый отчёт
# ============================================================
def _render_text_report(r: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"MIDI-АНАЛИЗ: {r['name']}")
    lines.append("=" * 60)
    lines.append(f"Время анализа:  {r['analyzed_at']}")
    lines.append(f"Длительность:   {r['duration_sec']} сек")
    lines.append(f"Темп:           {r['tempo_bpm']} BPM")
    lines.append(f"Инструментов:   {len(r['instruments'])}")
    lines.append("")

    lines.append("--- ОБЩАЯ СТАТИСТИКА ---")
    s = r["summary"]
    lines.append(f"Всего нот:      {s['total_notes']}")
    lines.append(f"Плотность:      {s['notes_per_second']} нот/сек")
    lines.append(
        f"Диапазон:       {s['pitch_min_name']} ({s['pitch_min']}) — "
        f"{s['pitch_max_name']} ({s['pitch_max']})"
    )
    lines.append(f"Размах:         {s['pitch_range_semitones']} полутонов")
    lines.append(f"Средняя нота:   {s['pitch_avg_name']} ({s['pitch_avg']})")
    lines.append(f"Медиана:        {s['pitch_median_name']} ({s['pitch_median']})")
    lines.append(
        f"Velocity:       min={s['velocity_min']}, "
        f"avg={s['velocity_avg']}, max={s['velocity_max']}"
    )
    lines.append("")

    lines.append("--- РАСПРЕДЕЛЕНИЕ ПО ОКТАВАМ ---")
    for key in sorted(r["octave_distribution"].keys(), key=lambda x: int(x)):
        info = r["octave_distribution"][key]
        lines.append(
            f"  Октава O{info['octave']}: {info['count']} нот "
            f"({info['percent']}%)"
        )
    lines.append("")

    lines.append("--- ТОП-10 НОТ ---")
    for item in r["top_10_notes"]:
        lines.append(f"  {item['note']}: {item['count']} раз")
    lines.append("")

    lines.append("--- ПО ИНСТРУМЕНТАМ ---")
    for instr in r["instruments"]:
        drums = " (drums)" if instr["is_drum"] else ""
        lines.append(
            f"  [{instr['program']}] {instr['name']}{drums}: "
            f"{instr['notes_count']} нот, "
            f"{instr['pitch_min_name']}–{instr['pitch_max_name']}, "
            f"velocity avg={instr['velocity_avg']}"
        )
    lines.append("=" * 60)
    return "\n".join(lines)