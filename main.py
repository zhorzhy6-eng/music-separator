"""
Точка входа AST.
Настраивает логирование, запускает UI.
"""
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы работали абсолютные импорты core.* и ui.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> int:
    setup_logging()
    logger.info("=== AST запуск ===")
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