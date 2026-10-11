"""
VerdictView — готовое представление вердикта супер-анализа для интерфейса.

Зачем: раньше окно само лазило во внутреннюю схему отчёта
(result["level_8_phase_smearing"]["phase_smearing_db"] и т.п.). Схема отчёта —
это внутреннее дело анализатора: стоило переименовать уровень или вложить поле
поглубже, как интерфейс показывал пустоту или падал.

Теперь UI получает вот эту плоскую коробочку с готовыми числами и о схеме
отчёта ничего не знает: меняется отчёт — правим только view_from_super_report.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerdictView:
    """Плоский набор показателей для окна результата."""

    kid: str = ""
    pro: str = ""
    bleed_pct: float = 0.0
    vpi_db: float = 0.0
    phase_db: float = 0.0
    tilt_pct: float = 0.0
    transient_pct: float = 0.0
    real_artifacts: int = 0
    musical_events: int = 0
    coverage_pct: float = 0.0
    # DECISION: два флага честности. Раньше окно показывало числа всегда, и на
    # минусовке (вокала в треке нет) человек читал «отпечаток голоса 139.8 дБ».
    # Теперь окно знает, когда измерять было нечего.
    vocal_present: bool = True     # есть ли в треке вокал вообще
    vpi_known: bool = True         # удалось ли посчитать отпечаток голоса

    @property
    def coverage_ok(self) -> bool:
        """Достаточно ли данных, чтобы вердикту можно было верить."""
        return self.coverage_pct >= 5.0


def _num(value, default: float = 0.0) -> float:
    """Аккуратно достаёт число: в отчёте может не быть поля или стоять None."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def view_from_super_report(report: dict) -> VerdictView:
    """
    Отчёт супер-анализа → VerdictView.

    Все обращения к схеме отчёта живут только здесь.
    """
    report = report or {}
    v = report.get("verdict", {}) or {}
    cov = report.get("coverage", {}) or {}
    vpi = report.get("level_6_vpi") or {}

    # Покрытие активных кадров считается в блоке VPI. В отчёте оно лежит во
    # вложенном разделе coverage (coverage.vpi), но понимаем и плоский вариант —
    # чтобы переезд поля не ломал окно.
    coverage_pct = None
    vpi_cov = cov.get("vpi", {}) if isinstance(cov.get("vpi"), dict) else {}
    for candidate in (cov.get("active_coverage_pct"),
                      vpi_cov.get("active_coverage_pct"),
                      vpi.get("active_coverage_pct")):
        if candidate is not None:
            coverage_pct = candidate
            break

    return VerdictView(
        kid=str(v.get("kid", "") or ""),
        pro=str(v.get("pro", "") or ""),
        bleed_pct=_num((report.get("level_5_bleed") or {}).get("bleed_pct")),
        vpi_db=_num(vpi.get("vpi_db")),
        phase_db=_num((report.get("level_8_phase_smearing") or {}).get("phase_smearing_db")),
        tilt_pct=_num((report.get("level_9_spectral_tilt") or {}).get("spectral_tilt_pct")),
        transient_pct=_num(
            (report.get("level_10_transient_loss") or {}).get("transient_loss_pct")),
        real_artifacts=int(_num((report.get("level_12_real_artifacts") or {}).get("count"))),
        musical_events=int(_num(
            (report.get("level_13_instrumental_events") or {}).get("count"))),
        coverage_pct=_num(coverage_pct),
        # В старых отчётах (до 11.10.2026) этих полей нет: считаем, что вокал был,
        # а VPI посчитан — иначе старые отчёты разом стали бы «пустыми».
        vocal_present=bool(v.get("vocal_present", True)),
        vpi_known=vpi.get("vpi_db") is not None,
    )
