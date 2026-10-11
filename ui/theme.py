"""
Оформление приложения: цвета, шрифты, скругления, отступы.

Зачем отдельный модуль: раньше цвета были рассыпаны по файлам интерфейса
(`#7f3f3f`, `#8e44ad`, кортежи вида `("#6a6a6a", "#9a9a9a")`), и светлая тема
получалась механической инверсией тёмной. Теперь всё оформление — в одном месте:

  * ПАЛИТРА DARK / LIGHT — подобранные цвета, а не «дефолт CustomTkinter»;
  * FONT_* — единая типографика (крупнее заголовок, мельче подпись);
  * RADIUS_* — скругления: кнопки 8, карточки 12, поля 6;
  * SPACE_* — отступы: внутри карточки 16, между карточками 12, между блоками 8.

Правило для остального кода: НЕ писать цвета и размеры шрифта строками на месте.
Кнопки берём из `ui/button_styles.py`, панели — из `ui/card.py`, а здесь читаем
готовые значения.

Важно: `get_theme()` по-прежнему возвращает обычный словарь — его читают и окно,
и компоненты, и тесты. Новые ключи (panel_hover, border, warning, danger…)
добавлены, старые (bg, panel, accent, text, text_dim, success, error) на месте.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# =============================================================
#  Цвета
# =============================================================
# Тёмная тема — основная: глубокий графит (не чёрный) и мягкий синий акцент.
DARK = {
    "bg": "#121212",            # фон окна
    "panel": "#1e1e1e",         # панели и карточки
    "panel_hover": "#252525",   # панель под курсором
    "border": "#2a2a2a",        # тонкие границы
    "accent": "#4A9EFF",        # главное действие
    "accent_hover": "#3A8EEF",
    "accent_dim": "#2F5C8F",    # приглушённый акцент: пунктир зоны файла
    "accent_soft": "#1B2C42",   # подложка выбранной строки и кнопок-обводок
    "text": "#E8E8E8",          # основной текст
    "text_dim": "#9A9A9A",      # второстепенный
    "text_muted": "#6A6A6A",    # третичный
    "success": "#34D399",
    "warning": "#FBBF24",
    "error": "#F87171",
    "danger": "#EF4444",        # «Отмена» и прочие опасные действия
    "danger_hover": "#DC2626",
    "on_accent": "#FFFFFF",     # текст на акцентной заливке
    "overlay": "#000000",       # «плёнка», закрывающая меню по клику мимо
}

# Светлая тема — не инверсия тёмной, а своя: белые карточки на мягком фоне,
# насыщенный синий акцент (на белом мягкий #4A9EFF читается плохо).
LIGHT = {
    "bg": "#F8F9FA",
    "panel": "#FFFFFF",
    "panel_hover": "#F3F4F6",
    "border": "#E5E7EB",
    "accent": "#2563EB",
    "accent_hover": "#1D4ED8",
    "accent_dim": "#93B4F5",
    "accent_soft": "#E8EFFD",   # подложка выбранной строки и кнопок-обводок
    "text": "#111827",
    "text_dim": "#6B7280",
    "text_muted": "#9CA3AF",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "danger": "#DC2626",
    "danger_hover": "#B91C1C",
    "on_accent": "#FFFFFF",
    "overlay": "#FFFFFF",
}


def get_theme(name: str = "dark") -> dict:
    """Тема по имени. Неизвестное имя — тёмная (она основная)."""
    return DARK if name != "light" else LIGHT


# =============================================================
#  Типографика
# =============================================================
# Размеры по ролям: заголовок окна 20, вкладки 15, карточки 13, кнопки 13,
# основной текст 12, подписи 11.
FONT_TITLE = 20
FONT_TAB = 15
FONT_CARD = 13
FONT_BUTTON = 13
FONT_BODY = 12
FONT_CAPTION = 11

# DECISION: Inter — если он установлен в системе; иначе Segoe UI (Windows),
# иначе Helvetica. Ставить шрифт файлом внутрь проекта не стали: лишние
# сотни килобайт и лицензионные тонкости ради того же результата.
FONT_CANDIDATES = ("Inter", "Inter Variable", "Inter Display", "Segoe UI", "Helvetica")
MONO_CANDIDATES = ("Consolas", "Cascadia Mono", "DejaVu Sans Mono", "Courier New")

_family: str | None = None
_mono_family: str | None = None
# Ссылки на созданные CTkFont: держим, чтобы Tk не собрал их сборщиком мусора
_font_cache: dict[tuple, object] = {}


def _pick_font(root, candidates: tuple[str, ...], fallback: str) -> str:
    """
    Первое доступное семейство из списка.

    `tkinter.font.families()` требует живого окна, поэтому вызывается уже при
    собранном интерфейсе: до этого момента работают значения по умолчанию.
    """
    try:
        from tkinter import font as tkfont

        available = {name.lower() for name in tkfont.families(root)}
    except Exception:
        logger.debug("Список шрифтов недоступен — беру запасной", exc_info=True)
        return fallback
    for name in candidates:
        if name.lower() in available:
            return name
    return fallback


def font_family(root=None) -> str:
    """Семейство основного шрифта (Inter → Segoe UI → Helvetica)."""
    global _family
    if _family is None and root is not None:
        _family = _pick_font(root, FONT_CANDIDATES, "Segoe UI")
        logger.info("Шрифт интерфейса: %s", _family)
    return _family or "Segoe UI"


def mono_family(root=None) -> str:
    """Семейство моноширинного шрифта — для таймера плеера."""
    global _mono_family
    if _mono_family is None and root is not None:
        _mono_family = _pick_font(root, MONO_CANDIDATES, "Courier New")
    return _mono_family or "Consolas"


def font_spec(size: int, weight: str = "normal", mono: bool = False):
    """
    Описание шрифта для виджета.

    Отдаём КОРТЕЖ, а не ctk.CTkFont: кортеж понимают и CustomTkinter, и обычные
    tk-виджеты (Canvas, Label подсказки), и его не нужно создавать через живой
    root. Это заметно проще, чем тащить ссылки на CTkFont через пол-интерфейса.
    """
    family = (mono_family() if mono else font_family())
    return (family, size, "bold") if weight == "bold" else (family, size)


# =============================================================
#  Скругления и отступы
# =============================================================
RADIUS_BUTTON = 8
RADIUS_CARD = 12
RADIUS_INPUT = 6
RADIUS_ROUND = 20        # круглые кнопки плеера

SPACE_CARD = 16          # внутри карточки
SPACE_BETWEEN = 12       # между карточками
SPACE_BLOCK = 8          # между блоками внутри карточки
SPACE_TIGHT = 4          # между связанными элементами

HEIGHT_BUTTON = 40       # главная кнопка
HEIGHT_BUTTON_SMALL = 36 # второстепенная
HEIGHT_INPUT = 32        # тихая кнопка / поле ввода
PLAYER_HEIGHT = 56       # высота плеера-карточки

WAVE_HEIGHT = 120        # волна с файлом
WAVE_PLACEHOLDER_HEIGHT = 80   # заглушка, пока файла нет
DROPZONE_HEIGHT = 140    # зона «перетащите файл»


# =============================================================
#  Простыми словами: расшифровка технических терминов
# =============================================================
# Один словарь на всё приложение: эти пояснения показываются в подсказках к
# настройкам и к отчётам. Термины пришли из аудиотехники, и человеку без
# музыкального образования они не говорят ничего — значит, надо объяснить.
PLAIN_WORDS = {
    "SDR": "Оценка чистоты разделения: чем выше, тем лучше отделён вокал.",
    "Bleed": "Протечка: сколько вокала попало в минус.",
    "VPI": "Отпечаток голоса в минусе: сколько голоса «просвечивает» "
           "через музыку.",
    "Phase Smearing": "Смазанность атаки: звук теряет чёткость начала нот.",
    "Spectral Tilt": "Затемнение верха: высокие частоты стали тише, "
                     "звук «глуше».",
    "Transient Loss": "Потеря чёткости ударов: барабаны и щипки звучат мягче, "
                      "чем в оригинале.",
    "dBFS": "Уровень громкости от нуля вниз: 0 — предел, −6 — вдвое тише.",
    "notch": "Узкий вырез частот: убирает лишний горб на одной частоте.",
    "shifts": "Сколько раз модель прогоняет трек со сдвигом: больше — чище, "
              "но дольше.",
    "overlap": "Перекрытие кусков при склейке: помогает убрать стыки.",
    "fp16": "Половинная точность расчёта: экономит видеопамять.",
    "batch": "Сколько кусков считается за один заход: больше — быстрее, "
             "но ест видеопамять.",
    "MIDI": "Файл с нотами: его открывают музыкальные редакторы.",
}

# Короткие пояснения к метрикам отчёта: имя → (как показать, что значит)
METRIC_HELP = (
    ("Остатки голоса в музыке", "Bleed",
     "Сколько вокала осталось в минусе. Меньше — лучше."),
    ("Отпечаток голоса в минусе", "VPI",
     "Насколько голос «просвечивает» через музыку. Меньше — лучше."),
    ("Размазанность звука", "Phase Smearing",
     "Потеря чёткости атаки. Меньше — лучше."),
    ("Затемнение верха", "Spectral Tilt",
     "Стали ли тише высокие частоты. Меньше — лучше."),
    ("Потеря чёткости ударов", "Transient Loss",
     "Насколько мягче звучат барабаны. Меньше — лучше."),
)


def plain(word: str) -> str:
    """Расшифровка термина; неизвестный термин возвращаем как есть."""
    return PLAIN_WORDS.get(word, "")


__all__ = [
    "DARK", "LIGHT", "get_theme",
    "FONT_TITLE", "FONT_TAB", "FONT_CARD", "FONT_BUTTON", "FONT_BODY",
    "FONT_CAPTION", "FONT_CANDIDATES", "MONO_CANDIDATES",
    "font_family", "mono_family", "font_spec",
    "RADIUS_BUTTON", "RADIUS_CARD", "RADIUS_INPUT", "RADIUS_ROUND",
    "SPACE_CARD", "SPACE_BETWEEN", "SPACE_BLOCK", "SPACE_TIGHT",
    "HEIGHT_BUTTON", "HEIGHT_BUTTON_SMALL", "HEIGHT_INPUT", "PLAYER_HEIGHT",
    "WAVE_HEIGHT", "WAVE_PLACEHOLDER_HEIGHT", "DROPZONE_HEIGHT",
    "PLAIN_WORDS", "METRIC_HELP", "plain",
]
