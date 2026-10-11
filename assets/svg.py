"""
Разбор SVG и отрисовка контуров средствами Pillow.

Зачем свой модуль: иконки Lucide — это SVG-контуры, а Pillow их не понимает.
Ставить cairosvg или svglib = тянуть новые зависимости в venv и ломать переносимый
запуск. Контуры Lucide простые (линии, окружности, кривые), поэтому парсер
занимает один небольшой файл и не требует ничего, кроме Pillow.

Где используется:
  * tools/make_icons.py — превращает скачанные SVG в PNG-иконки;
  * ui/components.py — рисует ПУНКТИРНУЮ рамку зоны файла (та же логика
    «обвести контур штрихами»).

Наружу нужны три вещи:
  * parse_path(d) → список подконтуров (каждый — список точек);
  * dashed_polyline(points, ...) → куски той же линии, разбитые на штрихи;
  * draw_strokes(...) → нарисовать контуры на картинке Pillow.

Единицы: система координат SVG (у Lucide — viewBox 0 0 24 24). Перевод в пиксели
делает вызывающий: он знает, во сколько раз увеличивает картинку.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from PIL import Image, ImageDraw

# Сколько точек на одну кривую при разбиении. 24 хватает с запасом: кривые
# Lucide — короткие, а при увеличении ×4 ошибка получается меньше полупикселя.
_CURVE_STEPS = 24
_ARC_STEPS = 24

# Разбор строки пути: команда + числа (включая «1.5e-3» и «.5»)
_TOKEN_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


@dataclass(frozen=True)
class StrokeStyle:
    """Как рисовать контур: цвет, толщина, закруглённые концы и стыки."""

    color: str = "#FFFFFF"
    width: float = 2.0
    cap_round: bool = True
    join_round: bool = True


# =============================================================
#  Разбор пути
# =============================================================
def parse_path(d: str) -> list[list[tuple[float, float]]]:
    """
    Строка атрибута `d` → список подконтуров из точек.

    Поддерживаются все команды SVG-путей: M/L/H/V (линии), C/S/Q/T (кривые),
    A (дуги), Z (замыкание). Кривые разбиваются на короткие отрезки, поэтому
    наружу всегда выходит ломаная — её просто рисовать и легко «пунктирить».
    """
    subs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    pos = (0.0, 0.0)          # текущая точка
    start = (0.0, 0.0)        # начало подконтура (для Z)
    prev_ctrl: tuple[float, float] | None = None   # для S и T
    prev_cmd = ""

    for cmd, raw in _commands(d):
        nums = [float(x) for x in raw]
        rel = cmd.islower()
        up = cmd.upper()

        if up == "M":
            for i in range(0, len(nums) - 1, 2):
                point = _abs(nums[i], nums[i + 1], pos, rel)
                if i == 0:
                    if len(current) > 1:
                        subs.append(current)
                    current = [point]
                    start = point
                else:
                    # Лишние пары после M трактуются как L (так велит стандарт)
                    current.append(point)
                pos = point
            prev_ctrl = None

        elif up == "L":
            for i in range(0, len(nums) - 1, 2):
                point = _abs(nums[i], nums[i + 1], pos, rel)
                current.append(point)
                pos = point
            prev_ctrl = None

        elif up == "H":
            for value in nums:
                x = pos[0] + value if rel else value
                pos = (x, pos[1])
                current.append(pos)
            prev_ctrl = None

        elif up == "V":
            for value in nums:
                y = pos[1] + value if rel else value
                pos = (pos[0], y)
                current.append(pos)
            prev_ctrl = None

        elif up in ("C", "S"):
            step = 6 if up == "C" else 4
            for i in range(0, len(nums) - step + 1, step):
                if up == "C":
                    c1 = _abs(nums[i], nums[i + 1], pos, rel)
                    c2 = _abs(nums[i + 2], nums[i + 3], pos, rel)
                    end = _abs(nums[i + 4], nums[i + 5], pos, rel)
                else:
                    # S: первая контрольная — отражение предыдущей
                    c1 = _reflect(pos, prev_ctrl)
                    c2 = _abs(nums[i], nums[i + 1], pos, rel)
                    end = _abs(nums[i + 2], nums[i + 3], pos, rel)
                current.extend(_cubic(pos, c1, c2, end))
                prev_ctrl = c2
                pos = end

        elif up in ("Q", "T"):
            step = 4 if up == "Q" else 2
            for i in range(0, len(nums) - step + 1, step):
                if up == "Q":
                    ctrl = _abs(nums[i], nums[i + 1], pos, rel)
                    end = _abs(nums[i + 2], nums[i + 3], pos, rel)
                else:
                    ctrl = _reflect(pos, prev_ctrl)
                    end = _abs(nums[i], nums[i + 1], pos, rel)
                current.extend(_quadratic(pos, ctrl, end))
                prev_ctrl = ctrl
                pos = end

        elif up == "A":
            for i in range(0, len(nums) - 6, 7):
                rx, ry = abs(nums[i]), abs(nums[i + 1])
                rot = nums[i + 2]
                large = bool(int(nums[i + 3]))
                sweep = bool(int(nums[i + 4]))
                end = _abs(nums[i + 5], nums[i + 6], pos, rel)
                current.extend(_arc(pos, rx, ry, rot, large, sweep, end))
                pos = end
            prev_ctrl = None

        elif up == "Z":
            if current:
                if current[0] != current[-1]:
                    current.append(current[0])
                subs.append(current)
                current = []
            pos = start
            prev_ctrl = None

        else:                                     # неизвестная команда — пропускаем
            prev_ctrl = None

        prev_cmd = up

    if len(current) > 1:
        subs.append(current)
    return subs


def parse_circle(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Окружность (элемент <circle>) → замкнутая ломаная."""
    return _ellipse_points(cx, cy, r, r)


def parse_rect(x: float, y: float, w: float, h: float, rx: float = 0.0,
               ry: float = 0.0) -> list[tuple[float, float]]:
    """Прямоугольник (<rect>), при rx/ry — со скруглёнными углами."""
    if rx <= 0 and ry <= 0:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    rx = rx or ry
    ry = ry or rx
    rx, ry = min(rx, w / 2), min(ry, h / 2)
    right, bottom = x + w, y + h
    k = 0.5523                       # коэффициент перевода дуги в кривую Безье

    points: list[tuple[float, float]] = []
    points.extend(_cubic((x + rx, y), (x + rx - rx * k, y), (x, y + ry - ry * k), (x, y + ry)))
    points.extend(_cubic((x, bottom - ry),
                         (x, bottom - ry + ry * k), (x + rx - rx * k, bottom), (x + rx, bottom)))
    points.extend(_cubic((right - rx, bottom),
                         (right - rx + rx * k, bottom), (right, bottom - ry + ry * k),
                         (right, bottom - ry)))
    points.extend(_cubic((right, y + ry),
                         (right, y + ry - ry * k), (right - rx + rx * k, y), (right - rx, y)))
    points.append(points[0])
    return points


def _commands(d: str):
    """Разбирает `d` на пары «команда — список чисел»."""
    tokens = _TOKEN_RE.findall(d or "")
    cmd: str | None = None
    nums: list[str] = []
    for letter, number in tokens:
        if letter:
            if cmd is not None:
                yield cmd, nums
            cmd, nums = letter, []
        else:
            nums.append(number)
    if cmd is not None:
        yield cmd, nums


def _abs(x: float, y: float, pos, rel: bool):
    return (pos[0] + x, pos[1] + y) if rel else (x, y)


def _reflect(pos, ctrl):
    """Отражение контрольной точки для гладких кривых (S/T)."""
    if ctrl is None:
        return pos
    return (2 * pos[0] - ctrl[0], 2 * pos[1] - ctrl[1])


def _cubic(p0, p1, p2, p3) -> list[tuple[float, float]]:
    """Кривая Безье третьего порядка → точки (без начальной)."""
    out = []
    for i in range(1, _CURVE_STEPS + 1):
        t = i / _CURVE_STEPS
        u = 1 - t
        out.append((
            u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1],
        ))
    return out


def _quadratic(p0, p1, p2) -> list[tuple[float, float]]:
    """Квадратичная кривая → точки (без начальной)."""
    out = []
    for i in range(1, _CURVE_STEPS + 1):
        t = i / _CURVE_STEPS
        u = 1 - t
        out.append((
            u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
            u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
        ))
    return out


def _ellipse_points(cx: float, cy: float, rx: float, ry: float,
                    steps: int = _ARC_STEPS * 2) -> list[tuple[float, float]]:
    out = []
    for i in range(steps + 1):
        a = 2 * math.pi * i / steps
        out.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return out


def _arc(p0, rx: float, ry: float, rot_deg: float, large: bool, sweep: bool,
         p1) -> list[tuple[float, float]]:
    """
    Дуга эллипса (команда A) → точки.

    Реализация по приложению F.6.5 стандарта SVG: сначала находим центр эллипса,
    потом идём по нему от начального угла к конечному. Без этого «облако» в
    иконке upload-cloud превратилось бы в прямую линию.
    """
    if rx == 0 or ry == 0 or p0 == p1:
        return [p1]

    phi = math.radians(rot_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    # Шаг 1: середина отрезка в системе координат эллипса
    dx2, dy2 = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1 = cos_p * dx2 + sin_p * dy2
    y1 = -sin_p * dx2 + cos_p * dy2

    # Шаг 2: радиусы не меньше, чем нужно для соединения точек
    lam = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if lam > 1:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale

    # Шаг 3: центр
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    factor = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        factor = -factor
    cxp = factor * rx * y1 / ry
    cyp = -factor * ry * x1 / rx
    cx = cos_p * cxp - sin_p * cyp + (p0[0] + p1[0]) / 2
    cy = sin_p * cxp + cos_p * cyp + (p0[1] + p1[1]) / 2

    # Шаг 4: начальный и конечный углы
    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        norm = math.hypot(ux, uy) * math.hypot(vx, vy)
        value = max(-1.0, min(1.0, dot / norm)) if norm else 1.0
        a = math.acos(value)
        return -a if (ux * vy - uy * vx) < 0 else a

    start = angle(1, 0, (x1 - cxp) / rx, (y1 - cyp) / ry)
    delta = angle((x1 - cxp) / rx, (y1 - cyp) / ry,
                  (-x1 - cxp) / rx, (-y1 - cyp) / ry)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi

    out = []
    for i in range(1, _ARC_STEPS + 1):
        t = start + delta * i / _ARC_STEPS
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        out.append((cos_p * ex - sin_p * ey + cx, sin_p * ex + cos_p * ey + cy))
    return out


# =============================================================
#  Пунктир
# =============================================================
def dashed_polyline(points, dash: float, gap: float,
                    phase: float = 0.0) -> list[list[tuple[float, float]]]:
    """
    Режет ломаную на штрихи: возвращает куски линии длиной dash с пропусками gap.

    `phase` сдвигает рисунок штрихов вдоль линии — за счёт этого пунктир зоны
    файла «бежит» при перетаскивании.
    """
    if len(points) < 2 or dash <= 0:
        return [list(points)]

    period = dash + max(gap, 0.001)
    out: list[list[tuple[float, float]]] = []
    piece: list[tuple[float, float]] = []
    travelled = 0.0

    def on_dash(position: float) -> bool:
        return ((position + phase) % period) < dash

    prev = points[0]
    for point in points[1:]:
        seg_len = math.hypot(point[0] - prev[0], point[1] - prev[1])
        if seg_len <= 0:
            prev = point
            continue
        steps = max(1, int(seg_len / 2) + 1)      # дробим отрезок, чтобы пунктир был ровным
        for i in range(1, steps + 1):
            frac = i / steps
            cur = (prev[0] + (point[0] - prev[0]) * frac,
                   prev[1] + (point[1] - prev[1]) * frac)
            position = travelled + seg_len * frac
            if on_dash(position):
                if not piece:
                    piece = [prev]
                piece.append(cur)
            elif piece:
                out.append(piece)
                piece = []
        travelled += seg_len
        prev = point

    if piece:
        out.append(piece)
    return out


def rounded_rect_path(x: float, y: float, w: float, h: float,
                      radius: float) -> list[tuple[float, float]]:
    """Замкнутая ломаная: прямоугольник со скруглёнными углами."""
    return parse_rect(x, y, w, h, rx=radius, ry=radius)


# =============================================================
#  Отрисовка
# =============================================================
def draw_strokes(size: tuple[int, int], subpaths, style: StrokeStyle,
                 scale: float = 1.0, dash: tuple[float, float] | None = None,
                 phase: float = 0.0) -> Image.Image:
    """
    Рисует контуры на прозрачной картинке и возвращает её.

    `scale` — во сколько раз увеличиваем систему координат SVG (для чёткой
    отрисовки рисуем крупно, потом уменьшаем). `dash` — (штрих, пропуск) для
    пунктира.
    """
    # DECISION: рисуем вчетверо крупнее и уменьшаем. Так края линий получаются
    # гладкими без ручного сглаживания: Pillow этого не умеет.
    ss = 4
    width, height = int(size[0] * ss), int(size[1] * ss)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    line_width = max(1, round(style.width * scale * ss))

    for points in subpaths:
        scaled = [(px * scale * ss, py * scale * ss) for px, py in points]
        parts = ([p for p in dashed_polyline(scaled, dash[0] * scale * ss,
                                             dash[1] * scale * ss, phase * scale * ss)]
                 if dash else [scaled])
        for part in parts:
            if len(part) < 2:
                continue
            draw.line(part, fill=style.color, width=line_width, joint="curve")
            if style.cap_round:
                # Pillow рисует концы обрубленными: добавляем кружки на стыках
                r = line_width / 2
                for px, py in (part[0], part[-1]):
                    draw.ellipse((px - r, py - r, px + r, py + r), fill=style.color)

    return img.reduce(ss)


__all__ = [
    "StrokeStyle", "parse_path", "parse_circle", "parse_rect",
    "dashed_polyline", "rounded_rect_path", "draw_strokes",
]
