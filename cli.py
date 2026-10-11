"""
Headless CLI: ast separate|transcribe|piano|analyze|super|engines|worker-check.

Зачем: без окна приложение нельзя ни протестировать автоматически, ни пустить
пачкой по списку файлов. CLI делает ровно то же, что кнопки в интерфейсе, но
через те же самые движки (core.engines) и тот же воркер (core.worker).

Примеры (из корня проекта):
    .\\.venv312\\Scripts\\python.exe cli.py engines
    .\\.venv312\\Scripts\\python.exe cli.py separate "track.wav" --engine "BS-Roformer: Duality-V2 ✅ баланс"
    .\\.venv312\\Scripts\\python.exe cli.py separate "track.wav" --engine bs:BS-Roformer-SW.ckpt --out output
    .\\.venv312\\Scripts\\python.exe cli.py transcribe output\\track_vocals.wav --out output\\track_vocals.mid
    .\\.venv312\\Scripts\\python.exe cli.py piano output\\track_instrumental.wav --out output\\track_piano.mid
    .\\.venv312\\Scripts\\python.exe cli.py analyze "track.wav"
    .\\.venv312\\Scripts\\python.exe cli.py super --vocals output\\t_vocals.wav --instrumental output\\t_instrumental.wav
"""
from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

# Корень проекта — от расположения файла, чтобы CLI работал из любой папки
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _setup() -> None:
    """Логирование в logs/ast.log — как у оконной версии."""
    _force_utf8_console()
    from core.logging_config import setup_logging
    setup_logging()


def _force_utf8_console() -> None:
    """
    Печатает в консоль UTF-8, а не в кодировке системы.

    В подписях движков есть эмодзи («🏆 инстр»), а консоль Windows по умолчанию
    работает в cp1251 — такая строка роняла CLI с UnicodeEncodeError. Опаснее
    всего то, что падал ГЛАВНЫЙ поток: рабочий поток оставался жив, интерпретатор
    начинал выключаться, и настоящая причина выглядела как загадочное
    «can't register atexit after shutdown» из недр librosa/joblib.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass            # не текстовый поток (перенаправлен) — не беда


def _print_result_stems(result) -> None:
    """Печатает пути стемов (StemSet ведёт себя как словарь)."""
    for name, path in result.items():
        print(f"{name:14s} {path}")


# ============================================================
#  Команды
# ============================================================
def cmd_engines(args) -> int:
    """Список движков из реестра — то, что показывает выпадающий список в окне."""
    from core.engines import DEFAULT_ENGINE_ID, ENGINES
    print(f"Всего движков: {len(ENGINES)}\n")
    for spec in ENGINES.values():
        mark = "  (по умолчанию)" if spec.id == DEFAULT_ENGINE_ID else ""
        print(f"{spec.id}\n    {spec.label}  [{spec.kind}]{mark}")
        if spec.companion:
            print(f"    гибрид: вокал {spec.companion} + минус {spec.model}")
    return 0


def cmd_separate(args) -> int:
    """Разделение через воркер (та же очередь и отмена, что в интерфейсе)."""
    from core.engines import get_engine, resolve_engine_id
    from core.paths import OUTPUT_DIR
    from core.worker import DoneEvent, ErrorEvent, ProgressEvent, Worker

    src = Path(args.input)
    if not src.exists():
        print(f"Файл не найден: {src}")
        return 2

    try:
        spec = get_engine(resolve_engine_id(args.engine))
    except KeyError as e:
        print(e)
        print("Посмотреть список: cli.py engines")
        return 2

    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    worker = Worker()
    task_id = worker.submit_separation(spec, str(src), str(out_dir))
    print(f"Задача {task_id}: {spec.label}")
    print(f"Вывод: {out_dir}")

    result = _wait_task(worker, task_id, args.quiet)
    if result is None:
        return 1
    print("Готово:")
    _print_result_stems(result)
    return 0


def cmd_transcribe(args) -> int:
    """Одноголосная транскрипция (torchcrepe) через воркер."""
    from core.worker import Worker
    src = Path(args.input)
    if not src.exists():
        print(f"Файл не найден: {src}")
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    worker = Worker()
    task_id = worker.submit_transcription(str(src), str(out), instrument=args.instrument)
    result = _wait_task(worker, task_id, args.quiet)
    if result is None:
        return 1
    print(f"MIDI: {result}")
    return 0


def cmd_piano(args) -> int:
    """Транскрипция в фортепиано (MuScriptor) через воркер."""
    from core.worker import Worker
    src = Path(args.input)
    if not src.exists():
        print(f"Файл не найден: {src}")
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    worker = Worker()
    task_id = worker.submit_piano(str(src), str(out), model_size=args.model)
    result = _wait_task(worker, task_id, args.quiet)
    if result is None:
        return 1
    print(f"Piano MIDI: {result}")
    return 0


def cmd_analyze(args) -> int:
    """Анализ аудио: отчёт и графики в logs/analysis."""
    from core.analyzer import analyze_audio
    from core.paths import ANALYSIS_DIR

    src = Path(args.input)
    if not src.exists():
        print(f"Файл не найден: {src}")
        return 2
    out_dir = Path(args.out) if args.out else ANALYSIS_DIR
    report = analyze_audio(str(src), str(out_dir))
    print(f"Отчёт: {out_dir}")
    if isinstance(report, dict):
        levels = report.get("levels", {})
        if levels:
            print(f"RMS: {levels.get('rms_db')} dBFS, Peak: {levels.get('peak_db')} dBFS")
    return 0


def cmd_super(args) -> int:
    """СУПЕР-АНАЛИЗ пары вокал+минус. Показывает то же, что окно результата."""
    from core.paths import ANALYSIS_DIR
    from core.reports import view_from_super_report
    from core.super_analyzer import super_analyze_pair

    vocals = Path(args.vocals)
    instr = Path(args.instrumental)
    for p in (vocals, instr):
        if not p.exists():
            print(f"Файл не найден: {p}")
            return 2

    out_dir = Path(args.out) if args.out else ANALYSIS_DIR
    report = super_analyze_pair(str(vocals), str(instr), str(out_dir))
    view = view_from_super_report(report)

    print(view.kid)
    print()
    print(f"Bleed:            {view.bleed_pct} %")
    print(f"VPI:              {view.vpi_db} dB")
    print(f"Phase Smearing:   {view.phase_db} dB")
    print(f"Spectral Tilt:    {view.tilt_pct} %")
    print(f"Transient Loss:   {view.transient_pct} %")
    print(f"Реальных артефактов: {view.real_artifacts}, "
          f"музыкальных событий: {view.musical_events}")
    print(f"Покрытие активного вокала: {view.coverage_pct} %")
    if not view.coverage_ok:
        print("⚠️ Покрытие низкое — вердикту стоит доверять с осторожностью")
    print(f"\nОтчёты: {out_dir}")
    return 0


def cmd_worker_check(args) -> int:
    """Быстрая проверка воркера: очереди, отмена, закрытие."""
    from core.worker import CPU_TASKS, GPU_TASKS, Worker
    w = Worker()
    print(f"GPU-очередь (по одной):     {', '.join(sorted(t.name for t in GPU_TASKS))}")
    print(f"CPU-очередь (по две):       {', '.join(sorted(t.name for t in CPU_TASKS))}")
    print(f"Занят сейчас: {w.is_busy}")
    w.shutdown()
    print("Воркер закрыт, дочерних процессов не осталось")
    return 0


# ============================================================
#  Общее
# ============================================================
def _wait_task(worker, task_id: int, quiet: bool):
    """
    Ждёт окончания задачи, печатая прогресс. Возвращает результат или None.

    Печать прогресса — только в CLI: окно рисует его само.

    Ожидание — ЦИКЛ, а не самовызов. Раньше пустая очередь (queue.Empty на
    каждом тике в 0.2 с) уходила в `except Exception: return _wait_task(...)`,
    то есть функция вызывала себя же. На длинном треке таких тиков набираются
    тысячи, и каждый оставлял свой кадр в стеке — CLI падал с RecursionError
    («переполнение стека») прямо посреди Demucs. Причём падал ГЛАВНЫЙ поток:
    рабочий оставался жив, интерпретатор начинал выключаться, и настоящая
    причина выглядела как загадочная ошибка из недр torch/librosa.
    """
    from core.worker import DoneEvent, ErrorEvent, ProgressEvent

    last_printed = -1
    try:
        while True:
            try:
                ev = worker.events.get(timeout=0.2)
            except queue.Empty:
                continue                  # задача ещё идёт — просто ждём дальше

            if isinstance(ev, ProgressEvent) and ev.task_id == task_id:
                pct = int(ev.progress * 100)
                if not quiet and pct != last_printed:
                    last_printed = pct
                    print(f"  [{pct:3d}%] {ev.stage}")
            elif isinstance(ev, DoneEvent) and ev.task_id == task_id:
                return ev.result
            elif isinstance(ev, ErrorEvent) and ev.task_id == task_id:
                if getattr(ev, "cancelled", False):
                    print("⏹ Отменено")
                else:
                    print(f"Ошибка: {ev.message}")
                return None
    except KeyboardInterrupt:
        print("\n⏹ Прерываю по Ctrl+C...")
        worker.cancel(task_id)
        return None
    finally:
        # DECISION: shutdown ровно один раз — на выходе из ожидания. В прежнем
        # варианте он выполнялся ещё и на каждом рекурсивном вызове: первый же
        # пустой тик очереди закрывал пулы воркера и отменял задачу, которая
        # в этот момент считалась.
        worker.shutdown(wait=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ast",
        description="AST без окна: разделение, транскрипция, анализ.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("engines", help="список движков из реестра")
    p.set_defaults(func=cmd_engines)

    p = sub.add_parser("separate", help="разделить трек на вокал и минус")
    p.add_argument("input", help="входной аудиофайл")
    p.add_argument("--engine", required=True,
                   help="id движка или подпись из интерфейса (см. cli.py engines)")
    p.add_argument("--out", default=None, help="куда положить результат (по умолчанию output/)")
    p.add_argument("--quiet", action="store_true", help="не печатать прогресс")
    p.set_defaults(func=cmd_separate)

    p = sub.add_parser("transcribe", help="транскрипция в MIDI (одноголосная)")
    p.add_argument("input")
    p.add_argument("--out", required=True, help="куда положить .mid")
    p.add_argument("--instrument", default="Фортепиано")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("piano", help="транскрипция в фортепиано (MuScriptor)")
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="small")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_piano)

    p = sub.add_parser("analyze", help="анализ аудио (отчёт + графики)")
    p.add_argument("input")
    p.add_argument("--out", default=None, help="каталог отчётов (по умолчанию logs/analysis)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("super", help="СУПЕР-АНАЛИЗ пары вокал+минус")
    p.add_argument("--vocals", required=True)
    p.add_argument("--instrumental", required=True)
    p.add_argument("--out", default=None, help="каталог отчётов (по умолчанию logs/analysis)")
    p.set_defaults(func=cmd_super)

    p = sub.add_parser("worker-check", help="проверка очередей воркера")
    p.set_defaults(func=cmd_worker_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    _setup()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Прервано пользователем")
        return 130


if __name__ == "__main__":
    sys.exit(main())
