"""
Список иконок: что рисуем и каких размеров.

Отдельный файл без единого импорта — чтобы его читали и приложение, и
`tools/make_icons.py`, который эти иконки готовит. Если добавить имя сюда и
запустить инструмент, иконка появится на диске; если оставить как есть —
`ui/icons.get_icon()` вернёт None, и кнопка покажет запасной символ.

Набор — по одному смыслу на действие (см. ui/app_window.py):
    scissors      — разделить трек
    music         — вокал, аудиофайл
    piano         — минус как пианино
    activity      — проверить качество звука
    sparkles      — проверить качество разделения
    folder        — открыть папку
    chart-column  — отчёты о качестве
    settings      — настройки
    play/pause/square — плеер
    volume-2      — громкость
    upload-cloud  — зона файла
    file-text     — ноты (MIDI)
    save          — сохранить настройки
    rotate-ccw    — сбросить настройки
    x             — отмена
    moon / sun    — тема
    info          — о программе
    log-out       — выход
    alert-triangle — предупреждение в зоне файла
"""

# Имена файлов Lucide (без расширения)
ICON_NAMES: tuple[str, ...] = (
    "activity",
    "alert-triangle",
    "chart-column",
    "file-text",
    "folder",
    "info",
    "log-out",
    "moon",
    "music",
    "pause",
    "piano",
    "play",
    "rotate-ccw",
    "save",
    "scissors",
    "settings",
    "sparkles",
    "square",
    "sun",
    "upload-cloud",
    "volume-2",
    "x",
)

# Размеры, которые реально встречаются в интерфейсе:
# 16 — значок в строке, 20 — обычная кнопка, 24 — кнопка-иконка и центр зоны
# файла, 32 — крупная иконка зоны файла.
ICON_SIZES: tuple[int, ...] = (16, 20, 24, 32)

# DECISION: запасной набор — ОДИН символ на действие, а не два-три, как раньше
# («⏹ Отмена», «📊 MIDI-анализ»). Если иконок нет (например, не запускали
# tools/make_icons.py), интерфейс остаётся читаемым и без картинок.
EMOJI_FALLBACK: dict[str, str] = {
    "scissors": "✂",
    "music": "♪",
    "piano": "🎹",
    "activity": "◉",
    "sparkles": "✦",
    "folder": "🗀",
    "chart-column": "▤",
    "settings": "⚙",
    "play": "▶",
    "pause": "⏸",
    "square": "⏹",
    "volume-2": "🔊",
    "upload-cloud": "⬆",
    "file-text": "▤",
    "save": "💾",
    "rotate-ccw": "↩",
    "x": "✕",
    "moon": "🌙",
    "sun": "☀",
    "info": "ℹ",
    "log-out": "🚪",
    "alert-triangle": "⚠",
}


__all__ = ["ICON_NAMES", "ICON_SIZES", "EMOJI_FALLBACK"]
