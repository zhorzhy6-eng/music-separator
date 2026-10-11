"""
Иконки для интерфейса: загрузка PNG и перекраска в цвет темы.

Как устроено:
  * на диске лежат БЕЛЫЕ PNG (assets/icons/<имя>_<размер>.png) — их готовит
    tools/make_icons.py из SVG библиотеки Lucide;
  * при загрузке иконка перекрашивается в нужный цвет: одна картинка на все
    темы и на все состояния кнопки, а не по файлу на каждый случай;
  * результат кешируется вместе с готовым ctk.CTkImage — перерисовка не тратит
    время на повторную загрузку.

Если файла нет (иконки не готовили) — возвращаем None. Кнопка в этом случае
показывает запасной символ (`emoji_fallback`), а не остаётся пустой.
"""
from __future__ import annotations

import logging
from pathlib import Path

import customtkinter as ctk
from PIL import Image

from ui.icons_catalog import EMOJI_FALLBACK, ICON_NAMES, ICON_SIZES

logger = logging.getLogger(__name__)

ICONS_DIR = Path(__file__).resolve().parent.parent / "assets" / "icons"

# Кеш готовых иконок: (имя, размер, цвет) → ctk.CTkImage
_cache: dict[tuple[str, int, str], ctk.CTkImage] = {}
# Уже предупреждали, что иконок нет — второй раз в лог не пишем
_warned = False
# Какое окно Tk породило картинки в кеше. Картинки Tk живут внутри своего
# интерпретатора: если окно закрыли, старые ссылки становятся мусором
# («image pyimage1 doesn't exist»). Поэтому при смене окна кеш сбрасываем.
_cache_root = None


def _bind_to_root(widget) -> None:
    """Сбрасывает кеш, если иконки запрашивает уже другое окно Tk."""
    global _cache_root
    try:
        root = widget.winfo_toplevel()
    except Exception:                                # noqa: BLE001
        return
    if _cache_root is not root:
        _cache.clear()
        _cache_root = root


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _tint(img: Image.Image, color: str) -> Image.Image:
    """
    Перекрашивает белую иконку в нужный цвет, сохраняя прозрачность.

    Цветной рисунок не «заливаем» сверху: белая линия становится цветной, а
    прозрачный фон остаётся прозрачным. Иначе получился бы цветной квадрат.
    """
    red, green, blue = _hex_to_rgb(color)
    alpha = img.getchannel("A")
    tinted = Image.new("RGBA", img.size)
    # Доступ к пикселям через load(): он работает одинаково на всех версиях
    # Pillow (getdata помечен устаревшим, get_flattened_data есть не везде).
    src = alpha.load()
    dst = tinted.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            dst[x, y] = (red, green, blue, src[x, y])
    return tinted


def get_icon(name: str, size: int = 20, color: str = "#E8E8E8", master=None):
    """
    Иконка для виджета или None, если её нет.

    Цвет — обычная строка '#RRGGBB'. Требует созданного окна Tk: картинку
    CustomTkinter делает через Tk. `master` — виджет, которому нужна иконка:
    по нему понимаем, не сменилось ли окно (см. _bind_to_root).
    """
    global _warned
    if master is not None:
        _bind_to_root(master)
    if name not in ICON_NAMES or not size:
        return None

    key = (name, size, color)
    if key in _cache:
        return _cache[key]

    # Округляем до ближайшего готового размера: рисовать «20.5 пикселя» смысла нет
    source_size = min(ICON_SIZES, key=lambda s: abs(s - size))
    path = ICONS_DIR / f"{name}_{source_size}.png"
    if not path.exists():
        if not _warned:
            logger.warning(
                "Иконок нет в %s — кнопки покажут запасные символы. "
                "Включить иконки: tools/make_icons.py", ICONS_DIR)
            _warned = True
        return None

    try:
        with Image.open(path) as raw:
            img = _tint(raw.convert("RGBA"), color)
        if img.size != (size, size):
            img = img.resize((size, size), Image.LANCZOS)
        icon = ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
    except Exception:
        logger.exception("Не удалось загрузить иконку %s", path.name)
        return None

    _cache[key] = icon
    return icon


def emoji_fallback(name: str) -> str:
    """Запасной символ вместо иконки (пустая строка, если и его нет)."""
    return EMOJI_FALLBACK.get(name, "")


def clear_cache() -> None:
    """Сбрасывает кеш иконок (нужно тестам и при смене размера окна)."""
    _cache.clear()


__all__ = ["ICONS_DIR", "get_icon", "emoji_fallback", "clear_cache"]
