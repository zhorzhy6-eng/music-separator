"""
Выбор модели: три понятных вопроса вместо списка из 26 названий.

Зачем переделано: раньше список группировался ПО АВТОРУ модели (Viperx,
Duality, Gabox, Becruily). Человеку без музыкального и технического образования
это не говорит ничего: «Viperx-1297» и «Gabox bleedless v2» — просто набор
букв, и выбрать между ними невозможно. Теперь:

  * сверху три большие кнопки-пресета — «что вы хотите сделать?». Нажатие
    выбирает подходящую модель, знать её имя не нужно;
  * ниже — полный список по ЗАДАЧАМ («Убрать вокал», «Минус как в оригинале»,
    «Только вокал»…). В списке крупно понятное название («Без протечек
    вокала») и мелко пояснение; техническое имя, SDR, видеопамять и скорость
    убраны в подсказку при наведении;
  * выбранная модель помечена галочкой, рекомендуемые — «рекомендуется».

Как устроено:
  * сам виджет — кнопка с названием выбранного движка;
  * по нажатию открывается отдельное окно: пресеты и группы со строками;
  * наружу видны только get()/set() с id движка — как у обычного комбобокса,
    поэтому остальному коду всё равно, как это нарисовано.

Оформление берётся из темы (ui/theme.py): сам селектор её не выбирает, а
получает готовой и перекрашивается методом `apply_theme`.

Значения id берутся из реестра core/engines/: интерфейс — витрина реестра,
своего списка моделей у него нет.
"""
from __future__ import annotations

import logging
import tkinter as tk
from typing import Callable

import customtkinter as ctk

from core.engines import (
    DEFAULT_ENGINE_ID,
    ENGINES,
    PRESETS,
    EngineSpec,
    Preset,
    engines_by_group,
    group_caption,
    group_order,
    preset_engine_id,
)
from ui.button_styles import icon_image
from ui.theme import (
    FONT_BODY,
    FONT_CAPTION,
    FONT_CARD,
    RADIUS_BUTTON,
    RADIUS_CARD,
    SPACE_BLOCK,
    SPACE_CARD,
    SPACE_TIGHT,
    font_spec,
    get_theme,
)
from ui.tooltip import ToolTip
from ui.widgets import restore_main_focus

logger = logging.getLogger(__name__)

POPUP_WIDTH = 620
POPUP_MARGIN = 8
EMPTY_LABEL = "— движок не выбран —"
SELECTED_MARK = "✅"
RECOMMENDED_MARK = "рекомендуется"
EXPAND_MORE = "▸ Показать все модели (для профи)"
EXPAND_LESS = "▾ Скрыть список моделей"

# Пояснение к сокращениям прямо в списке: без него «SDR ≈ 12.98» ничего не значит
LEGEND = ("Не знаете, что выбрать — нажмите одну из кнопок выше: модель "
          "подберётся сама. Технические детали (имя модели, SDR, видеопамять) "
          "показываются при наведении мыши.")
PRESET_HEADER = "Что вы хотите сделать?"
ALL_MODELS_HEADER = "Все модели по задачам"


class EngineSelector(ctk.CTkFrame):
    """
    Выбор движка: пресеты-задачи сверху, полный список моделей по кнопке.

    get() возвращает id движка ("bs:...ckpt"), set() принимает id.
    """

    def __init__(self, master, on_change: Callable[[str], None] | None = None,
                 width: int = 340, theme: dict | None = None, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.on_change = on_change
        self._width = width
        self.theme = theme or get_theme("dark")
        self._engine_id: str | None = None
        self._popup: ctk.CTkToplevel | None = None
        self._overlay = None
        self._tips: list[ToolTip] = []
        self._expanded = False
        self._build()

    # ---------- сборка ----------
    def _build(self) -> None:
        theme = self.theme
        img = icon_image("settings", 16, theme["text"], master=self)
        self.button = ctk.CTkButton(
            self, text=f"{EMPTY_LABEL}  ▾", width=self._width, anchor="w",
            command=self.toggle,
            font=font_spec(FONT_BODY),
            fg_color=theme["panel_hover"], hover_color=theme["border"],
            text_color=theme["text"], corner_radius=RADIUS_BUTTON, height=36,
            image=img, compound="left" if img is not None else "text",
        )
        self.button.pack(side="left")
        ToolTip(self.button, self._button_tip)

    def apply_theme(self, theme: dict) -> None:
        """Перекрашивает кнопку селектора под новую тему."""
        self.theme = theme
        try:
            self.button.configure(fg_color=theme["panel_hover"],
                                  hover_color=theme["border"],
                                  text_color=theme["text"],
                                  image=icon_image("settings", 16, theme["text"],
                                                   master=self.button))
        except Exception:
            logger.debug("Селектор движков не перекрасился", exc_info=True)

    def _button_tip(self) -> str:
        """Подсказка на самой кнопке: что сейчас выбрано и за что оно отвечает."""
        if self._engine_id is None:
            return "Модель не выбрана. Нажмите, чтобы выбрать по задаче."
        spec = ENGINES.get(self._engine_id)
        if spec is None:
            return "Модель из настроек не найдена в реестре."
        return spec.tooltip + "\n\nНажмите, чтобы выбрать другую модель."

    # ---------- внешнее API ----------
    def get(self) -> str:
        """id выбранного движка. Если ничего не выбрано — движок по умолчанию."""
        return self._engine_id or DEFAULT_ENGINE_ID

    def values(self) -> list[str]:
        """
        Подписи всех движков — в том порядке, в каком они в списке.

        Нужно смоук-проверкам: они сверяют витрину с реестром, как раньше
        сверяли список CTkOptionMenu.
        """
        out: list[str] = []
        for specs in engines_by_group().values():
            out.extend(spec.label for spec in specs)
        return out

    def label(self) -> str:
        """
        Техническая подпись выбранного движка (или пустая строка).

        Осталась ради настроек, CLI и тестов: в settings.json лежит именно
        label, поэтому менять его на «понятное» имя нельзя.
        """
        spec = ENGINES.get(self._engine_id) if self._engine_id else None
        return spec.label if spec else ""

    def display_label(self) -> str:
        """Понятное название выбранного движка — то, что видно на кнопке."""
        spec = ENGINES.get(self._engine_id) if self._engine_id else None
        return spec.display_name if spec else ""

    def set(self, engine_id: str) -> None:
        """
        Устанавливает движок по id.

        Нужно и при старте (движок из настроек), и после смены настроек.
        Неизвестный id — не падаем: показываем «не выбран», а разбираться
        будет тот, кто читает get().
        """
        if engine_id not in ENGINES:
            logger.warning("Движок %r не найден в реестре — сбрасываю выбор", engine_id)
            self._engine_id = None
            self._update_button()
            return
        self._engine_id = engine_id
        self._update_button()

    def _update_button(self) -> None:
        spec = ENGINES.get(self._engine_id) if self._engine_id else None
        label = spec.display_name if spec else EMPTY_LABEL
        self.button.configure(text=f"{label}  ▾")

    # ---------- выпадающее окно ----------
    def toggle(self) -> None:
        """Открыть список или закрыть, если он уже открыт."""
        if self._popup is not None:
            self.close()
        else:
            self.open()

    def open(self) -> None:
        if self._popup is not None:
            return

        theme = self.theme
        popup = ctk.CTkToplevel(self)
        popup.wm_overrideredirect(True)
        popup.wm_attributes("-topmost", True)
        # Панель с рамкой и скруглением: список не должен сливаться с окном
        popup.configure(fg_color=theme["border"])
        self._popup = popup

        surface = ctk.CTkFrame(popup, fg_color=theme["panel"], corner_radius=10)
        surface.pack(fill="both", expand=True, padx=1, pady=1)
        self._surface = surface

        header = ctk.CTkLabel(surface, text=PRESET_HEADER,
                              font=font_spec(FONT_CARD, "bold"), anchor="w",
                              text_color=theme["text"])
        header.pack(fill="x", padx=SPACE_CARD, pady=(SPACE_CARD, 0))

        ctk.CTkLabel(surface, text=LEGEND, justify="left", anchor="w",
                     font=font_spec(FONT_CAPTION), wraplength=POPUP_WIDTH - 30,
                     text_color=theme["text_dim"]).pack(
            fill="x", padx=SPACE_CARD, pady=(2, SPACE_BLOCK))

        # DECISION: пресеты — статичные кнопки-карточки, а список моделей
        # сворачивается. Человеку, который просто хочет «убрать вокал», не
        # нужны 26 строк: они только пугают. Профи раскрывает список одной
        # кнопкой, и выбор остаётся в том же окне — окно не перерисовывается.
        for preset in PRESETS:
            self._build_preset(surface, preset)

        # Порядок важен: кнопка «Показать все модели» идёт ВЫШЕ списка. Пока
        # список не раскрыт, контейнер пуст и места не занимает — окно выглядит
        # как три пресета, а кнопка стоит сразу под ними. Когда список раскрыт,
        # он забирает всё оставшееся место, получает собственную прокрутку, а
        # кнопка становится подписью над ним («Скрыть список моделей»).
        self._toggle_row = ctk.CTkButton(
            surface, text=EXPAND_MORE, anchor="w", height=32,
            font=font_spec(FONT_BODY),
            fg_color="transparent", hover_color=theme["panel_hover"],
            text_color=theme["accent"], corner_radius=RADIUS_BUTTON,
            command=self._toggle_list,
        )
        self._toggle_row.pack(fill="x", padx=SPACE_BLOCK, pady=(SPACE_TIGHT, 0))
        # DECISION: у контейнера явная высота. Пустой CTkFrame с expand=True
        # растягивается на всё свободное место, и свёрнутое окно получалось бы
        # в пол-экрана с пустотой внизу. Ноль — это «списка нет», дальше высоту
        # задаёт _fill_list по содержимому.
        self._list_holder = ctk.CTkFrame(surface, fg_color="transparent", height=0)
        self._list_holder.pack(fill="both", expand=True)
        self._list_holder.pack_propagate(False)
        if self._expanded:
            self._fill_list(update_geometry=False)

        popup.update_idletasks()
        self._place_popup()
        self._show_overlay()
        popup.bind("<Escape>", lambda e: self.close())
        try:
            popup.focus_force()
        except Exception:
            logger.debug("Фокус на список движков не перешёл", exc_info=True)

    def _build_preset(self, parent, preset: Preset) -> None:
        """
        Кнопка-пресет: понятное название, под ним пояснение.

        Это Frame с двумя подписями, а не CTkButton с текстом в две строки:
        CTkButton рисует многострочный текст по центру, и колонка названий
        «пляшет». Здесь название прижато влево и читается как список.
        """
        theme = self.theme
        card = ctk.CTkFrame(parent, fg_color=theme["panel_hover"],
                            corner_radius=RADIUS_CARD)
        card.pack(fill="x", padx=SPACE_CARD, pady=SPACE_TIGHT)

        title = f"{preset.icon}  {preset.title}" if preset.icon else preset.title
        title_label = ctk.CTkLabel(card, text=title, anchor="w",
                                   font=font_spec(FONT_BODY, "bold"),
                                   text_color=theme["text"])
        title_label.pack(fill="x", padx=SPACE_CARD, pady=(SPACE_BLOCK, 0))
        subtitle_label = ctk.CTkLabel(card, text=preset.subtitle, anchor="w",
                                      font=font_spec(FONT_CAPTION),
                                      text_color=theme["text_dim"])
        subtitle_label.pack(fill="x", padx=SPACE_CARD, pady=(0, SPACE_TIGHT))

        for widget in (card, title_label, subtitle_label):
            widget.bind("<Button-1>", lambda e, p=preset: self._apply_preset(p))
            try:
                widget.configure(cursor="hand2")
            except Exception:
                logger.debug("Курсор-«рука» не поставился", exc_info=True)

        chosen = preset_engine_id(preset)
        spec = ENGINES.get(chosen) if chosen else None
        tip = f"Выберет модель «{spec.display_name}»" if spec else "Модель не найдена"
        ToolTip(card, f"{preset.title}\n{preset.subtitle}\n\n{tip}")

    def _toggle_list(self) -> None:
        """Раскрывает или сворачивает полный список моделей."""
        self._expanded = not self._expanded
        if self._expanded:
            self._fill_list(update_geometry=True)
            self._toggle_row.configure(text=EXPAND_LESS)
        else:
            self._clear_list()
            self._toggle_row.configure(text=EXPAND_MORE)
            self._popup.update_idletasks()
            self._place_popup()

    def _clear_list(self) -> None:
        for tip in self._tips:
            tip.destroy()
        self._tips.clear()
        for child in self._list_holder.winfo_children():
            child.destroy()
        # Контейнер снова «пустой»: высота в ноль, иначе свёрнутое окно
        # осталось бы в пол-экрана с пустотой на месте списка.
        self._list_holder.configure(height=0)

    def _fill_list(self, update_geometry: bool = True) -> None:
        """
        Строит полный список: группы-задачи и строки моделей внутри них.

        update_geometry — пересчитать ли размер окна. При открытии окна это
        делает open() после сборки; при раскрытии списка по кнопке — здесь.
        """
        theme = self.theme
        ctk.CTkLabel(self._list_holder, text=ALL_MODELS_HEADER, anchor="w",
                     font=font_spec(FONT_CARD, "bold"),
                     text_color=theme["text_dim"]).pack(
            fill="x", padx=SPACE_CARD, pady=(SPACE_BLOCK, SPACE_TIGHT))

        # DECISION: список прокручивается. 26 моделей в шести группах выше
        # экрана ноутбука — без прокрутки нижние группы были бы недостижимы.
        body = ctk.CTkScrollableFrame(self._list_holder, fg_color="transparent",
                                      width=POPUP_WIDTH - 40, height=100,
                                      scrollbar_button_color=theme["border"],
                                      scrollbar_button_hover_color=theme["text_muted"])
        body.pack(fill="both", expand=True, padx=SPACE_BLOCK, pady=(0, SPACE_BLOCK))
        self._body = body

        grouped = engines_by_group()
        for group in group_order():
            specs = grouped.get(group, [])
            if not specs:
                continue
            ctk.CTkLabel(body, text=group, anchor="w",
                         font=font_spec(FONT_CARD, "bold"),
                         text_color=theme["accent"]).pack(
                fill="x", padx=6, pady=(SPACE_BLOCK, 1))
            caption = group_caption(group)
            if caption:
                ctk.CTkLabel(body, text=caption, anchor="w", justify="left",
                             wraplength=POPUP_WIDTH - 70,
                             font=font_spec(FONT_CAPTION),
                             text_color=theme["text_dim"]).pack(
                    fill="x", padx=6, pady=(0, 2))
            for spec in specs:
                self._build_row(body, spec)

        body.update_idletasks()
        height = self._body_height(body)
        body.configure(height=height)
        self._list_holder.configure(height=height + self._list_header_height())

        if update_geometry and self._popup is not None:
            # DECISION: размер окна пересчитывается дважды с update() между
            # ними. Tk перестраивает геометрию не сразу: после первого
            # пересчёта окно ещё помнит прежнюю высоту, и раскрытый список
            # остался бы обрезанным до следующего открытия.
            for _ in range(2):
                self._popup.update_idletasks()
                self._place_popup()
                self._popup.update()

    def _list_header_height(self) -> int:
        """Высота заголовка «Все модели по задачам» вместе с его отступами."""
        for child in self._list_holder.winfo_children():
            if isinstance(child, ctk.CTkLabel):
                child.update_idletasks()
                return child.winfo_reqheight() + SPACE_BLOCK + SPACE_TIGHT
        return 0

    def _body_height(self, body: ctk.CTkScrollableFrame) -> int:
        """
        Высота прокручиваемого списка: сколько осталось от экрана.

        Считать её по содержимому нельзя: у CTkScrollableFrame
        winfo_reqheight() возвращает высоту ВСЕГО содержимого (для 26 моделей
        это под две тысячи пикселей). Поэтому берём свободное место на экране
        и вычитаем всё, что занято пресетами, кнопкой и заголовками.

        420 — предел «на глаз»: выше список уже не читается как выпадающий,
        и его начинает закрывать собственная кнопка на некоторых экранах.
        """
        screen_h = body.winfo_screenheight()
        # Пресеты, заголовки, кнопка-селектор и отступы — фиксированная часть.
        chrome = 190
        return int(max(200, min(420, screen_h - chrome - self._list_header_height())))

    def _build_row(self, parent, spec: EngineSpec) -> None:
        """
        Одна строка списка: понятное название крупно, описание мелко.

        Техническое имя оставлено в строке-подписи: по нему модель ищут в
        интернете, и оно же показано в настройках. Но глаз за него не цепляется —
        он серый и мелкий.
        """
        theme = self.theme
        selected = spec.id == self._engine_id
        name = f"{SELECTED_MARK}  {spec.display_name}" if selected else spec.display_name
        if spec.recommended:
            name = f"{name}  ·  {RECOMMENDED_MARK}"

        row = ctk.CTkButton(
            parent, text=name, anchor="w", height=30,
            font=font_spec(FONT_BODY, "bold" if selected else "normal"),
            fg_color=theme["accent_soft"] if selected else "transparent",
            hover_color=theme["panel_hover"],
            text_color=theme["accent"] if selected else theme["text"],
            corner_radius=RADIUS_BUTTON,
            command=lambda s=spec: self._choose(s.id),
        )
        row.pack(fill="x", padx=4, pady=(1, 0))

        # Подпись под названием: сначала понятное описание, потом — служебное
        # (техническое имя и скорость) в скобках. Техническое имя нужно, чтобы
        # сверить модель с настройками и с логом, но глаз за него не цепляется:
        # оно серое и мелкое.
        tail = f"{spec.label} · {spec.speed_hint}" if spec.speed_hint else spec.label
        line = f"{spec.plain_desc}   ({tail})" if spec.plain_desc else tail
        ctk.CTkLabel(parent, text=line, anchor="w", justify="left",
                     wraplength=POPUP_WIDTH - 80,
                     font=font_spec(FONT_CAPTION),
                     text_color=theme["text_dim"]).pack(
            fill="x", padx=10, pady=(0, 2))

        # DECISION: подсказка на каждой строке создаётся сразу (26 объектов) —
        # это дешевле, чем подменять текст «на лету», и исключает пустые
        # подсказки при быстром движении мыши по списку.
        self._tips.append(ToolTip(row, spec.tooltip, delay_ms=350))

    def _place_popup(self) -> None:
        """
        Ставит список под кнопкой; не влез вниз — раскрывает вверх.

        Высота считается от содержимого, но жёстко ограничивается экраном.
        Без ограничения раскрытый список из 26 моделей заказывал высоту
        больше монитора, и нижние строки становились недостижимы: полосу
        прокрутки видно, а дотянуться до неё нельзя.
        """
        popup = self._popup
        if popup is None:
            return
        popup.update_idletasks()
        w = max(popup.winfo_reqwidth(), POPUP_WIDTH)
        screen_w, screen_h = popup.winfo_screenwidth(), popup.winfo_screenheight()
        h = int(min(popup.winfo_reqheight(), screen_h - 2 * POPUP_MARGIN))

        x = self.button.winfo_rootx()
        y = self.button.winfo_rooty() + self.button.winfo_height() + 2
        if y + h > screen_h - POPUP_MARGIN:
            y = max(POPUP_MARGIN, self.button.winfo_rooty() - h - 2)
        x = max(POPUP_MARGIN, min(x, screen_w - w - POPUP_MARGIN))
        popup.wm_geometry(f"{w}x{h}+{int(x)}+{int(y)}")

    def _show_overlay(self) -> None:
        """
        «Плёнка» на главном окне: клик мимо списка его закрывает.

        Без неё список оставался бы висеть поверх интерфейса, пока не выберешь
        модель, — а это раздражает сильнее, чем лишний клик.
        """
        try:
            root = self.winfo_toplevel()
            overlay = tk.Frame(root, background=self.theme["overlay"], bd=0,
                               highlightthickness=0)
            overlay.place(x=0, y=0, relwidth=1, relheight=1)
            overlay.bind("<Button-1>", lambda e: self.close())
            self._overlay = overlay
        except Exception:
            logger.debug("Плёнка под списком движков не создалась", exc_info=True)

    def _apply_preset(self, preset: Preset) -> None:
        """
        Кнопка-пресет: выбирает модель под задачу и раскрывает её группу.

        Список раскрывается намеренно: человек видит, ЧТО именно выбралось и
        какие есть соседние варианты, — иначе модель менялась бы «вслепую».
        """
        engine_id = preset_engine_id(preset)
        if engine_id is None:
            logger.warning("У пресета «%s» нет доступных моделей", preset.title)
            return
        self._expanded = True
        self._choose(engine_id)

    def _choose(self, engine_id: str) -> None:
        changed = engine_id != self._engine_id
        self.set(engine_id)
        self.close()
        if changed and self.on_change is not None:
            try:
                self.on_change(engine_id)
            except Exception:
                logger.exception("Обработчик смены движка упал")

    def close(self) -> None:
        """Закрывает список и убирает всё, что он за собой оставил."""
        for tip in self._tips:
            tip.destroy()
        self._tips.clear()

        if self._overlay is not None:
            try:
                self._overlay.destroy()
            except Exception:
                logger.debug("Плёнка под списком уже удалена", exc_info=True)
            self._overlay = None

        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                logger.debug("Список движков уже закрыт", exc_info=True)
            self._popup = None

        # DECISION: без этого окно остаётся без клавиатуры. Всплывающий список
        # забирал фокус на себя, а после destroy() Tk его не возвращает:
        # focus_get() → None, и горячие клавиши плеера (→ ← Space Home End)
        # перестают работать, пока не щёлкнешь по окну. См. ui/widgets.py.
        restore_main_focus(self)

    @property
    def is_open(self) -> bool:
        """Открыт ли список (нужно тестам и смоук-проверкам)."""
        return self._popup is not None

    @property
    def is_expanded(self) -> bool:
        """Раскрыт ли полный список моделей (нужно тестам)."""
        return self._expanded


__all__ = ["EngineSelector"]
