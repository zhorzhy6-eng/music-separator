"""
Проверки покрытия супер-анализа.

Покрытие отвечает на вопрос «можно ли верить вердикту»: если активного вокала
в треке мало, метрики считаются по долям секунды. В отчёте должны быть поля
покрытия, а окно (через core.reports) — показывать их человеку.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
import pytest

from core.reports import view_from_super_report
from core.super_analyzer import _vocal_activity, super_analyze_pair

SR = 44100


def _write_pair(tmp_path, sec: float = 6.0):
    """Пара вокал+минус: вокал поёт громко, но с паузами (чтобы покрытие было не 100%)."""
    n = int(sec * SR)
    t = np.arange(n, dtype=np.float64) / SR
    vocals = np.zeros(n, dtype=np.float64)
    # «поём» через одну секунду из трёх
    for k in range(0, n, 3 * SR):
        vocals[k:k + SR] = 0.5 * np.sin(2 * np.pi * 440 * t[k:k + SR])
    instr = 0.2 * np.sin(2 * np.pi * 220 * t)

    v_path = tmp_path / "cov_vocals.wav"
    i_path = tmp_path / "cov_instrumental.wav"
    sf.write(str(v_path), vocals.astype(np.float32), SR, subtype="PCM_16")
    sf.write(str(i_path), instr.astype(np.float32), SR, subtype="PCM_16")
    return v_path, i_path


def test_vocal_activity_counts_coverage():
    """
    Покрытие = доля кадров заметно громче медианы.

    Порог адаптивный (не ниже 75-го перцентиля), поэтому берём три уровня
    громкости: тихо / средне / громко. Активными должны стать громкие кадры,
    а не какие-то случайные 25%.
    """
    rms_db = np.array([-60.0] * 50 + [-30.0] * 25 + [-10.0] * 25)
    active, quiet, info = _vocal_activity(rms_db)

    assert info["active_frames_total"] == 100
    assert info["active_frames_used"] == int(np.sum(active))
    assert info["active_frames_used"] >= 25, "громкие кадры не признаны активными"
    assert 0.0 < info["active_coverage_pct"] <= 100.0
    assert info["active_threshold_db"] is not None
    assert info["vocal_median_db"] is not None
    # тихими считаются самые тихие кадры
    assert int(np.sum(quiet)) >= 50
    assert len(active) == 100 and len(quiet) == 100


def test_vocal_activity_empty_input_is_safe():
    active, quiet, info = _vocal_activity(np.array([]))
    assert info["active_coverage_pct"] == 0.0
    assert info["active_frames_used"] == 0
    assert len(active) == 0


def test_report_contains_coverage_fields(tmp_path):
    """В отчёте есть блок coverage и в нём — проценты покрытия."""
    v_path, i_path = _write_pair(tmp_path)
    report = super_analyze_pair(str(v_path), str(i_path), str(tmp_path / "reports"))

    cov = report.get("coverage")
    assert isinstance(cov, dict), "в отчёте нет блока coverage"
    assert "vpi" in cov and "transient" in cov
    assert "active_coverage_pct" in cov["vpi"]
    assert 0.0 <= cov["vpi"]["active_coverage_pct"] <= 100.0
    # покрытие считается и посекундно
    assert "per_second" in cov

    # Ожидаемые уровни отчёта на месте (UI читает их через core.reports)
    for key in ("level_5_bleed", "level_6_vpi", "verdict",
                "level_12_real_artifacts", "level_13_instrumental_events"):
        assert key in report, f"в отчёте нет {key}"


def test_view_exposes_coverage_to_ui(tmp_path):
    """VerdictView отдаёт покрытие в окно — UI в схему отчёта не лезет."""
    v_path, i_path = _write_pair(tmp_path)
    report = super_analyze_pair(str(v_path), str(i_path), str(tmp_path / "reports"))

    view = view_from_super_report(report)
    assert view.coverage_pct == pytest.approx(
        report["coverage"]["vpi"]["active_coverage_pct"], abs=0.01)
    assert isinstance(view.kid, str)
    assert view.real_artifacts >= 0


def test_view_is_robust_to_missing_levels():
    """Отчёт старого формата (без coverage) не должен ронять окно."""
    view = view_from_super_report({"verdict": {"kid": "старый отчёт"}})
    assert view.kid == "старый отчёт"
    assert view.coverage_pct == 0.0
    assert view.coverage_ok is False


def test_view_accepts_flat_coverage():
    """Плоское поле покрытия тоже понимаем (схема отчёта может переехать)."""
    view = view_from_super_report({"coverage": {"active_coverage_pct": 42.5}})
    assert view.coverage_pct == pytest.approx(42.5)
    assert view.coverage_ok is True


# ============================================================
#  Регрессия QA-прогона 11.10.2026
# ============================================================
def _write_no_vocal_pair(tmp_path, sec: float = 6.0):
    """
    Пара, где вокала нет: вокал-стем — тишина, минус — обычная музыка.

    Так выглядит разделение, когда на вход подали готовый минус. На прогоне
    11.10.2026 вердикт для такого случая звучал как приговор: «голосовой
    отпечаток заметен (VPI 139.8 dB)», «минус темнеет (-65.1%)»,
    «подозрительных секунд: 88» — всё посчитано от тишины.
    """
    n = int(sec * SR)
    t = np.arange(n, dtype=np.float64) / SR
    vocals = np.zeros(n, dtype=np.float64)          # ровная тишина
    instr = 0.3 * np.sin(2 * np.pi * 220 * t)       # музыка

    v_path = tmp_path / "novocal_vocals.wav"
    i_path = tmp_path / "novocal_instrumental.wav"
    sf.write(str(v_path), vocals.astype(np.float32), SR, subtype="PCM_16")
    sf.write(str(i_path), instr.astype(np.float32), SR, subtype="PCM_16")
    return v_path, i_path


def test_track_without_vocals_gets_honest_verdict(tmp_path):
    """Нет вокала — вердикт говорит об этом, а не пугает числами от тишины."""
    v_path, i_path = _write_no_vocal_pair(tmp_path)
    report = super_analyze_pair(str(v_path), str(i_path), str(tmp_path / "reports"))

    verdict = report["verdict"]
    assert verdict.get("vocal_present") is False, (
        "отчёт не заметил, что вокала в треке нет")
    assert "вокала практически нет" in verdict["kid"]
    # и никаких «отпечатков голоса» в вердикте быть не должно
    assert "отпечаток" not in verdict["kid"].lower()

    view = view_from_super_report(report)
    assert view.vocal_present is False
    assert "вокала практически нет" in view.kid


def test_vpi_not_invented_when_reference_is_silence(tmp_path):
    """
    Опора сравнения — ровные нули: VPI не считается, а не выдаёт сотни дБ.

    Кадры, где вокал тихий, обязаны попадать на участки, где и минус молчит:
    тогда средний уровень опоры уходит в математический ноль.
    """
    n = int(8.0 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    vocals = np.full(n, 1e-6, dtype=np.float64)
    instr = np.zeros(n, dtype=np.float64)
    # «поём» короткими вспышками; всё остальное время оба файла почти молчат
    for k in range(0, n, 4 * SR):
        vocals[k:k + int(0.3 * SR)] = 0.5 * np.sin(2 * np.pi * 440 * t[k:k + int(0.3 * SR)])
        instr[k:k + int(0.3 * SR)] = 0.3 * np.sin(2 * np.pi * 220 * t[k:k + int(0.3 * SR)])

    v_path = tmp_path / "silref_vocals.wav"
    i_path = tmp_path / "silref_instrumental.wav"
    sf.write(str(v_path), vocals.astype(np.float32), SR, subtype="PCM_16")
    sf.write(str(i_path), instr.astype(np.float32), SR, subtype="PCM_16")

    report = super_analyze_pair(str(v_path), str(i_path), str(tmp_path / "reports"))
    vpi = report["level_6_vpi"]

    if "vpi_db" in vpi:
        # если посчитался — значение обязано быть физически осмысленным
        assert -60.0 < vpi["vpi_db"] < 60.0, (
            f"VPI {vpi['vpi_db']} dB — таких значений в природе не бывает")
    else:
        assert vpi.get("vpi_note"), "VPI не посчитан, но причина не указана"
        assert "VPI" in report["verdict"]["pro"]
        assert "не посчитан" in report["verdict"]["pro"]

    # окно должно узнать, что числа нет, а не показать ноль
    view = view_from_super_report(report)
    assert view.vpi_known == ("vpi_db" in vpi)
