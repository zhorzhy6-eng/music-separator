r"""
Полная проверка «как в жизни»: три движка + анализ + супер-анализ.

Запуск из корня проекта:
    .\.venv312\Scripts\python.exe tools\e2e_check.py [входной_файл]

По умолчанию берётся короткий клип logs\qa_refclip_10s.wav (10 секунд).
Всё пишется в logs\_stagecheck\e2e: пользовательские файлы не трогаются.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engines import get_engine, make_separator  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402
from core.reports import view_from_super_report  # noqa: E402
from core.stems import StemSet  # noqa: E402

DEFAULT_CLIP = PROJECT_ROOT / "logs" / "qa_refclip_10s.wav"
OUT_ROOT = PROJECT_ROOT / "logs" / "_stagecheck" / "e2e"

ENGINES = [
    "demucs:htdemucs",                                  # Demucs
    "bs:model_bs_roformer_ep_317_sdr_12.9755.ckpt",      # BS-Roformer (модель уже скачана)
    "hybrid:htdemucs_ft+model_bs_roformer_ep_317_sdr_12.9755.ckpt",   # гибрид
]

results: list[tuple[str, bool, str]] = []


def run_engine(engine_id: str, src: Path) -> None:
    label = engine_id.split(":")[0]
    out_dir = OUT_ROOT / label
    print(f"\n=== {engine_id} ===")
    t0 = time.monotonic()
    try:
        sep = make_separator(get_engine(engine_id))
        result = sep.separate(str(src), str(out_dir))
        if not isinstance(result, StemSet):
            raise AssertionError(f"вернулся не StemSet, а {type(result).__name__}")
        vocals = result["vocals"]
        instr = result["instrumental"]
        ok = vocals.exists() and instr.exists()
        print(f"    vocals:       {vocals.name} ({vocals.stat().st_size // 1024} КБ)")
        print(f"    instrumental: {instr.name} ({instr.stat().st_size // 1024} КБ)")
        results.append((engine_id, ok, f"{time.monotonic() - t0:.1f} с"))
    except Exception as e:
        print(f"    ОШИБКА: {type(e).__name__}: {e}")
        results.append((engine_id, False, str(e)[:80]))


def run_analysis(vocals: Path, instr: Path) -> None:
    """Анализ и супер-анализ по готовой паре — как кнопки в интерфейсе."""
    from core.analyzer import analyze_audio
    from core.super_analyzer import super_analyze_pair

    report_dir = OUT_ROOT / "reports"
    print("\n=== анализ аудио (вокал) ===")
    rep = analyze_audio(str(vocals), str(report_dir))
    print(f"    RMS {rep['levels']['rms_db']} dBFS, peak {rep['levels']['peak_db']} dBFS")

    print("\n=== СУПЕР-АНАЛИЗ пары ===")
    report = super_analyze_pair(str(vocals), str(instr), str(report_dir))
    view = view_from_super_report(report)
    print(f"    {view.kid}")
    print(f"    bleed {view.bleed_pct}% | VPI {view.vpi_db} dB | "
          f"tilt {view.tilt_pct}% | покрытие {view.coverage_pct}%")
    results.append(("анализ + супер-анализ", view.coverage_pct > 0,
                    f"покрытие {view.coverage_pct}%"))


def main() -> int:
    setup_logging()
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLIP
    if not src.exists():
        print(f"Нет входного файла: {src}")
        return 2
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Вход: {src}")

    for engine_id in ENGINES:
        run_engine(engine_id, src)

    # Анализ — на паре от Demucs (самый быстрый движок)
    demucs_dir = OUT_ROOT / "demucs"
    base = src.stem
    vocals = demucs_dir / f"{base}_vocals.wav"
    instr = demucs_dir / f"{base}_instrumental.wav"
    if vocals.exists() and instr.exists():
        run_analysis(vocals, instr)

    print("\n================ ИТОГ ================")
    all_ok = True
    for name, ok, note in results:
        print(f"{'OK  ' if ok else 'ПЛОХО'} {name:55s} {note}")
        all_ok = all_ok and ok
    print("=====================================")
    print("ОТЧЁТЫ:", OUT_ROOT / "reports")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
