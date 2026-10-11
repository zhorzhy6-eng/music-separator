"""
Точка входа AST.
Настраивает логирование, запускает UI.
"""
import logging
import sys
from pathlib import Path

# Корень проекта — от расположения этого файла, а не от текущего каталога,
# поэтому main.py можно запускать из любой папки
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Запуск без консоли (pythonw.exe из AST.bat) ---
# DECISION: этот блок стоит ДО импорта core.* и ui.* — намеренно.
# У pythonw.exe, запущенного двойным щелчком (без консоли), sys.stdout и
# sys.stderr равны None. Тогда падает первое же обращение к ним: полоска tqdm
# внутри audio-separator пишет в sys.stderr, и BS-разделение обрывалось с
# «'NoneType' object has no attribute 'write'», хотя окно уже открылось.
# Подменяем «немые» потоки файлами в logs/: писать есть куда, и после сбоя
# остаётся след, по которому видно причину. buffering=1 (построчная запись) —
# чтобы хвост лога не потерялся, если процесс упадёт.
if sys.stdout is None or sys.stderr is None:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = open(log_dir / "stdout.log", "a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = open(log_dir / "stderr.log", "a", encoding="utf-8", buffering=1)

from core.logging_config import setup_logging
from core.paths import ANALYSIS_DIR, LOGS_DIR, OUTPUT_DIR, PROJECT_ROOT as PATHS_ROOT

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging()
    logger.info(
        "=== AST запуск ===",
        extra={
            "ctx_project_root": str(PATHS_ROOT),
            "ctx_output_dir": str(OUTPUT_DIR),
            "ctx_logs_dir": str(LOGS_DIR),
            "ctx_analysis_dir": str(ANALYSIS_DIR),
            "ctx_cwd": str(Path.cwd()),
        },
    )
    try:
        from ui.app_window import AppWindow
        app = AppWindow()
        app.mainloop()
    except Exception:
        logger.exception("Фатальная ошибка в UI")
        return 1
    logger.info("=== AST завершение ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
