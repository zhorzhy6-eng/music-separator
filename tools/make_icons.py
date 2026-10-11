r"""
Иконки Lucide: скачать SVG и превратить в PNG для интерфейса.

Запуск (из корня проекта):

    .\.venv312\Scripts\python.exe tools\make_icons.py

Что делает:
  1. скачивает SVG нужных иконок из библиотеки Lucide (ISC, бесплатно);
  2. кладёт их в assets/icons/svg/ — чтобы иконки можно было потом перерисовать
     без интернета, а не гадать, откуда взялась картинка;
  3. рисует из них белые PNG нужных размеров в assets/icons/.

Белый цвет — нарочно: `ui/icons.py` перекрашивает иконку в цвет темы при
загрузке (одна картинка на все темы, а не по файлу на каждый цвет).

Флаги:
    --force   перекачать SVG даже если файл уже есть
    --check   ничего не качать, только проверить, что все PNG на месте

Интернет нужен один раз. Если сети нет — приложение работает без иконок:
`ui/icons.py` вернёт None, а кнопка останется с текстом (см. _EMOJI_FALLBACK).
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from assets.svg import (  # noqa: E402
    StrokeStyle,
    draw_strokes,
    parse_circle,
    parse_path,
    parse_rect,
)
from ui.icons_catalog import ICON_NAMES, ICON_SIZES  # noqa: E402

ICONS_DIR = PROJECT_ROOT / "assets" / "icons"
SVG_DIR = ICONS_DIR / "svg"

# Откуда берём: версия зафиксирована (1.55.0), чтобы картинки не «поехали» со
# временем. Второй адрес — запасной (тот же набор, другой сайт).
SOURCES = (
    "https://raw.githubusercontent.com/lucide-icons/lucide/1.55.0/icons/{name}.svg",
    "https://unpkg.com/lucide-static@1.55.0/icons/{name}.svg",
)

# Крупный размер отрисовки: 24 единицы SVG в 96 пикселей (×4). UI потом
# уменьшает до нужного размера — так линии остаются гладкими.
RENDER_SIZE = 96
VIEWBOX = 24.0
STROKE_WIDTH = 2.0                       # как в самих иконках Lucide

_ELEMENT_RE = re.compile(
    r"<(path|circle|rect)\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')


def download(name: str, force: bool = False) -> Path | None:
    """Скачивает SVG иконки. Уже есть — не трогаем (если не --force)."""
    target = SVG_DIR / f"{name}.svg"
    if target.exists() and not force:
        return target
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    for template in SOURCES:
        url = template.format(name=name)
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  не вышло с {url.split('/')[2]}: {e}")
            continue
        target.write_text(data, encoding="utf-8")
        return target
    return None


def render(svg_text: str, size: int) -> "object":
    """SVG → картинка Pillow нужного размера (белые линии, прозрачный фон)."""
    subpaths: list[list[tuple[float, float]]] = []
    for element, attrs_text in _ELEMENT_RE.findall(svg_text):
        attrs = dict(_ATTR_RE.findall(attrs_text))
        kind = element.lower()
        if kind == "path" and attrs.get("d"):
            subpaths.extend(parse_path(attrs["d"]))
        elif kind == "circle":
            subpaths.append(parse_circle(
                float(attrs.get("cx", 0)), float(attrs.get("cy", 0)),
                float(attrs.get("r", 0))))
        elif kind == "rect":
            subpaths.append(parse_rect(
                float(attrs.get("x", 0)), float(attrs.get("y", 0)),
                float(attrs.get("width", 0)), float(attrs.get("height", 0)),
                float(attrs.get("rx", 0) or 0), float(attrs.get("ry", 0) or 0)))

    if not subpaths:
        raise ValueError("в SVG не нашлось ни одного контура")

    scale = RENDER_SIZE / VIEWBOX
    img = draw_strokes((RENDER_SIZE, RENDER_SIZE), subpaths,
                       StrokeStyle(color="#FFFFFF", width=STROKE_WIDTH),
                       scale=scale)
    if size != RENDER_SIZE:
        from PIL import Image
        img = img.resize((size, size), Image.LANCZOS)
    return img


def png_path(name: str, size: int) -> Path:
    return ICONS_DIR / f"{name}_{size}.png"


def check() -> int:
    """Проверяет, что все нужные PNG на месте (для тестов и смоук-проверок)."""
    missing = [png_path(name, size).name
               for name in ICON_NAMES for size in ICON_SIZES
               if not png_path(name, size).exists()]
    if missing:
        print(f"НЕТ {len(missing)} файлов, например: {', '.join(missing[:6])}")
        print("Запустите: .\\.venv312\\Scripts\\python.exe tools\\make_icons.py")
        return 1
    print(f"OK: {len(ICON_NAMES) * len(ICON_SIZES)} PNG на месте "
          f"({len(ICON_NAMES)} иконок, размеры: {', '.join(map(str, ICON_SIZES))})")
    return 0


def main() -> int:
    # Консоль Windows по умолчанию в cp1251 и падает на «×» и «⚠» — печатаем в UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                            # noqa: BLE001
            pass

    parser = argparse.ArgumentParser(description="Иконки Lucide → PNG")
    parser.add_argument("--force", action="store_true", help="перекачать SVG")
    parser.add_argument("--check", action="store_true", help="только проверить")
    args = parser.parse_args()

    if args.check:
        return check()

    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for index, name in enumerate(ICON_NAMES, start=1):
        svg_file = download(name, force=args.force)
        if svg_file is None:
            failed.append(name)
            print(f"[{index}/{len(ICON_NAMES)}] {name}: НЕ СКАЧАЛСЯ")
            continue
        try:
            text = svg_file.read_text(encoding="utf-8")
            for size in ICON_SIZES:
                render(text, size).save(png_path(name, size))
        except Exception as e:                       # noqa: BLE001
            failed.append(name)
            print(f"[{index}/{len(ICON_NAMES)}] {name}: ошибка отрисовки: {e}")
            continue
        print(f"[{index}/{len(ICON_NAMES)}] {name}: " +
              ", ".join(f"{s}px" for s in ICON_SIZES))

    print()
    if failed:
        print(f"Не получилось: {len(failed)} — {', '.join(failed)}")
        print("Приложение запустится и без них: у кнопок останется текст.")
        return 1
    print(f"Готово: {len(ICON_NAMES)} иконок, размеры: "
          f"{', '.join(map(str, ICON_SIZES))} — в {ICONS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
