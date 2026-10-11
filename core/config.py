"""
Конфигурация приложения.

Читается из config/settings.json. Значения по умолчанию зашиты в код,
поэтому отсутствие или порча файла приложение не роняет.

Путь к файлу берётся из core.paths (от корня проекта), а не из рабочего каталога.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

from core.paths import CONFIG_DIR

logger = logging.getLogger(__name__)

SETTINGS_FILE = CONFIG_DIR / "settings.json"


@dataclass
class AppConfig:
    """Настройки, которые читает интерфейс и тракт разделения."""

    default_engine: str = "BS-Roformer: Duality-V2 ✅ баланс"
    demucs_device: str = "cuda"
    demucs_shifts: int = 1
    demucs_overlap: float = 0.25
    notch_enabled: bool = False
    target_peak: float = 0.891
    output_sample_rate: int = 44100
    keep_temp_dirs: bool = False

    # --- BS-Roformer через Python-API (core/engines/bs_engine_api.py) ---
    # DECISION: три поля, а не десяток. Это ровно то, что нельзя было настроить
    # в subprocess-версии: точность счёта, размер батча и «держать модель
    # в памяти». Всё остальное у модели берётся из её же конфига.
    bs_use_fp16: bool = False          # половинная точность (GPU); на GTX 1660 быстрее НЕ будет
    bs_batch_size: int = 0             # 0 — как в конфиге модели (обычно 1)
    bs_keep_model_loaded: bool = True  # держать модель в памяти между треками (~10 с экономии)

    @classmethod
    def load(cls) -> "AppConfig":
        """
        Читает настройки. Если файла нет — работает на значениях по умолчанию
        и пишет файл, чтобы его можно было править руками.

        Важно: существующий файл НЕ перезаписывается. Иначе любое чтение затирало бы
        пользовательские правки (и шумно логировало ошибку, если файл только для чтения).
        """
        if not SETTINGS_FILE.exists():
            cfg = cls()
            try:
                cfg.save()
                logger.info("Создан файл настроек", extra={"ctx_file": str(SETTINGS_FILE)})
            except Exception:
                logger.exception(
                    "Не удалось создать settings.json — работаю на значениях по умолчанию"
                )
            return cfg

        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            # Неизвестные ключи игнорируем, отсутствующие берём из значений по умолчанию
            known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
            cfg = cls(**known)
            logger.info("Настройки загружены", extra={"ctx_file": str(SETTINGS_FILE)})
            return cfg
        except Exception:
            logger.exception("Не удалось прочитать settings.json — работаю на значениях по умолчанию")
            return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# Настройки читаются один раз за запуск и кэшируются: они влияют на параметры
# разделения, а не на каждый отдельный стем. Поменял settings.json — перезапусти
# приложение.
_cached_config: AppConfig | None = None


def get_config(reload: bool = False) -> AppConfig:
    """Текущие настройки приложения (из config/settings.json)."""
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = AppConfig.load()
    return _cached_config
