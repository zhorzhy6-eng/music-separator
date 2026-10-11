"""
Единая точка путей проекта.

Корень считается от расположения этого файла (__file__), а НЕ от рабочего
каталога: приложение можно запускать из любой папки, пути не «уедут».
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
ANALYSIS_DIR = LOGS_DIR / "analysis"
ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"

# Каталоги, без которых приложение не стартует, создаём сразу при импорте
for _d in (OUTPUT_DIR, LOGS_DIR, ANALYSIS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
