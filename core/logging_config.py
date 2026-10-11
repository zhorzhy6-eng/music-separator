"""
Централизованная настройка логирования.
Формат: JSON-lines (удобно парсить и грепать).
Логи пишутся в logs/ast.log и дублируются в консоль.
"""
import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

# Папка для логов рядом с проектом
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "ast.log"


class JsonFormatter(logging.Formatter):
    """Форматтер: каждая запись — одна строка JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        # Если в лог передан exception — добавим traceback
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Дополнительные поля (extra=...)
        for k, v in record.__dict__.items():
            if k.startswith("ctx_"):
                payload[k[4:]] = v
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Вызывается один раз из main.py.
    Настраивает корневой логгер: файл + консоль.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Убираем возможные дубли при повторном вызове
    for h in list(root.handlers):
        root.removeHandler(h)

    # --- Файловый handler с ротацией (5 МБ × 3 файла) ---
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    # --- Консольный handler (человекочитаемый) ---
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(console)

    logging.getLogger(__name__).info(
        "Логирование инициализировано", extra={"ctx_logfile": str(LOG_FILE)}
    )