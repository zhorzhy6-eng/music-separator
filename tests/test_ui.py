"""
Проверки окна («официанта»): состояние кнопок и разбор отчёта.

Окно поднимается по-настоящему, но скрытым и без запуска разделения. Если в
системе нет графики (например, запуск по ssh), тест пропускается, а не падает.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import customtkinter as ctk

from core.worker import DoneEvent, ErrorEvent, TaskType
from ui.viewmodels import UiState

tkinter = pytest.importorskip("tkinter")


def _make_window():
    """Собирает окно. Нет графики — тест пропускается."""
    try:
        from ui.app_window import AppWindow
        app = AppWindow()
    except Exception as e:                      # pragma: no cover
        pytest.skip(f"графика недоступна: {e}")
    app.withdraw()
    return app


def _buttons(app) -> dict:
    return {
        "separate": app.btn_separate,
        "analyze": app.btn_analyze,
        "voc_midi": app.btn_voc_midi,
        "inst_midi": app.btn_inst_midi,
        "piano": app.btn_piano,
        "super": app.btn_super_pair,
        "midi_analyze": app.btn_analyze_midi,
        "cancel": app.btn_cancel,
    }


def _states(app) -> dict:
    return {name: btn.cget("state") for name, btn in _buttons(app).items()}


# ---------- UiState ----------
def test_reset_for_new_file_clears_old_results(tmp_path):
    """Новый трек — старые пути сбрасываются (иначе работа уйдёт на чужой файл)."""
    st = UiState(source=tmp_path / "old.wav", vocals=tmp_path / "old_vocals.wav",
                 instrumental=tmp_path / "old_inst.wav", midi=tmp_path / "old.mid",
                 busy=True)

    st.reset_for_new_file(tmp_path / "new.wav")

    assert st.source == tmp_path / "new.wav"
    assert st.vocals is None
    assert st.instrumental is None
    assert st.midi is None
    assert st.busy is False


# ---------- Кнопки ----------
def test_start_state_has_everything_disabled():
    """Пока файл не выбран, работать не с чем."""
    app = _make_window()
    try:
        states = _states(app)
        assert states["separate"] == "disabled"
        assert states["cancel"] == "disabled"
    finally:
        app.destroy()


def test_load_file_enables_separate(tmp_path):
    """Загрузили файл — можно разделять; остальное ждёт результата разделения."""
    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        app._load_file(str(wav))
        states = _states(app)
        assert states["separate"] == "normal"
        assert states["analyze"] == "normal"
        assert states["piano"] == "disabled"
        assert states["super"] == "disabled"
    finally:
        app.destroy()


def test_busy_disables_actions_and_enables_cancel(tmp_path):
    """Пока идёт работа — кнопки запуска выключены, «Отмена» доступна."""
    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        app._load_file(str(wav))
        app.current_task_id = 1
        app._set_busy(True)

        states = _states(app)
        assert states["separate"] == "disabled"
        assert states["analyze"] == "disabled"
        assert states["cancel"] == "normal"
    finally:
        app.destroy()


def test_cancel_button_disables_itself():
    """После нажатия «Отмена» кнопка сразу гаснет, чтобы не жать её дважды."""
    app = _make_window()
    try:
        app.current_task_id = 1
        app._set_busy(True)
        assert app.btn_cancel.cget("state") == "normal"

        app._on_cancel()

        assert app.btn_cancel.cget("state") == "disabled"
    finally:
        app.destroy()


def test_buttons_restore_from_disk_after_error(tmp_path, monkeypatch):
    """
    После ошибки кнопки включаются не «слепо», а по факту: если для текущего
    трека есть вокал и минус — соответствующие кнопки снова доступны.
    """
    stem = "song"
    wav = tmp_path / f"{stem}.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")
    for kind in ("vocals", "instrumental"):
        sf.write(str(tmp_path / f"{stem}_{kind}.wav"),
                 np.zeros(1000, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        # Кнопки смотрят в output/ проекта — подменяем каталог на временный
        app.output_dir = tmp_path
        app._load_file(str(wav))
        app._set_busy(True)
        monkeypatch.setattr("ui.app_window.messagebox.showerror",
                            lambda *a, **k: None)

        app._handle_event(ErrorEvent(task_id=1, message="что-то упало"))

        states = _states(app)
        assert states["piano"] == "normal", "минус есть, а кнопка выключена"
        assert states["voc_midi"] == "normal", "вокал есть, а кнопка выключена"
        assert states["super"] == "normal"
        assert states["cancel"] == "disabled"
        assert app.ui_state.busy is False
    finally:
        app.destroy()


def test_cancel_event_does_not_scare_user(tmp_path, monkeypatch):
    """Отмена — не ошибка: красное окно с ошибкой не показываем."""
    app = _make_window()
    try:
        shown = []
        monkeypatch.setattr("ui.app_window.messagebox.showerror",
                            lambda *a, **k: shown.append(a))
        app._set_busy(True)
        app._handle_event(ErrorEvent(task_id=1, message="⏹ Отменено", cancelled=True))

        assert shown == [], "отмена показана как ошибка"
        assert app.ui_state.busy is False
    finally:
        app.destroy()


# ---------- Разбор отчёта ----------
def test_super_report_is_rendered_without_touching_schema(tmp_path, monkeypatch):
    """
    Окно показывает результат супер-анализа по готовому представлению
    (core.reports), не разбирая схему отчёта самостоятельно.
    """
    app = _make_window()
    try:
        shown = []
        monkeypatch.setattr("ui.app_window.messagebox.showinfo",
                            lambda title, text, *a, **k: shown.append(text))

        report = {
            "verdict": {"kid": "✅ Всё хорошо", "pro": "детали"},
            "level_5_bleed": {"bleed_pct": 1.5},
            "level_6_vpi": {"vpi_db": -2.0},
            "level_8_phase_smearing": {"phase_smearing_db": 0.3},
            "level_9_spectral_tilt": {"spectral_tilt_pct": 4.0},
            "level_10_transient_loss": {"transient_loss_pct": 2.0},
            "level_12_real_artifacts": {"count": 3},
            "level_13_instrumental_events": {"count": 1},
            "coverage": {"vpi": {"active_coverage_pct": 21.5}},
        }
        app._set_busy(True)
        app._handle_event(DoneEvent(task_id=1, task_type=TaskType.ANALYZE_PAIR,
                                    result=report))

        assert len(shown) == 1
        text = shown[0]
        assert "✅ Всё хорошо" in text
        assert "1.5%" in text          # bleed
        assert "21.5%" in text         # покрытие
        assert app.ui_state.busy is False
    finally:
        app.destroy()


def test_low_coverage_is_warned_about(tmp_path, monkeypatch):
    """Маленькое покрытие — предупреждаем: вердикту верить нельзя."""
    app = _make_window()
    try:
        shown = []
        monkeypatch.setattr("ui.app_window.messagebox.showinfo",
                            lambda title, text, *a, **k: shown.append(text))
        report = {"verdict": {"kid": "вердикт"},
                  "coverage": {"vpi": {"active_coverage_pct": 1.2}}}
        app._handle_event(DoneEvent(task_id=1, task_type=TaskType.ANALYZE_PAIR,
                                    result=report))
        assert "ненадёжным" in shown[0]
    finally:
        app.destroy()


# ---------- Смоук ----------
def test_engine_list_comes_from_registry():
    """Список движков в окне — тот же, что в реестре (один источник правды)."""
    from core.engines import all_labels

    app = _make_window()
    try:
        assert sorted(app.engine_combo.values()) == sorted(all_labels())
        assert app.engine_combo.label() in all_labels()
    finally:
        app.destroy()


# ---------- Вкладки ----------
def test_window_has_four_tabs():
    """Вкладки есть, и они именно те, что задуманы."""
    from ui.app_window import TAB_ANALYZE, TAB_SEPARATE, TAB_SETTINGS, TAB_TRANSCRIBE

    app = _make_window()
    try:
        assert list(app.tabs._tab_dict.keys()) == [
            TAB_SEPARATE, TAB_TRANSCRIBE, TAB_ANALYZE, TAB_SETTINGS]
    finally:
        app.destroy()


def test_settings_menu_switches_to_settings_tab():
    """«⚙ Настройки…» из меню открывает вкладку настроек, а не окно поверх окна."""
    from ui.app_window import TAB_SETTINGS

    app = _make_window()
    try:
        app._go_to_settings()
        assert app.tabs.get() == TAB_SETTINGS
    finally:
        app.destroy()


def test_progress_bar_hidden_until_task_runs(tmp_path):
    """Пустой прогресс-бар не занимает место: он появляется только с задачей."""
    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        assert app.progress_row.winfo_manager() == "", "полоса видна без задачи"

        app._load_file(str(wav))
        assert app.progress_row.winfo_manager() == "", "полоса видна без задачи"

        app.current_task_id = 1
        app._set_busy(True)
        assert app.progress_row.winfo_manager() == "pack", "полосы нет во время задачи"

        app._set_busy(False)
        assert app.progress_row.winfo_manager() == "", "полоса осталась после задачи"
    finally:
        app.destroy()


def test_empty_wave_shows_placeholder():
    """Пока файла нет, вместо пустых осей 0.0–1.0 — заглушка с подсказкой."""
    from ui.app_window import WAVE_PLACEHOLDER

    app = _make_window()
    try:
        assert app.wave._last_path is None
        assert not app.wave.ax.axison, "у заглушки остались оси"
        # Текст заглушки нарисован прямо на графике
        drawn = [t.get_text() for t in app.wave.ax.texts]
        assert WAVE_PLACEHOLDER in drawn, drawn
    finally:
        app.destroy()


def test_analysis_source_falls_back_to_stems_on_disk(tmp_path):
    """
    Плеер пуст, а стемы на диске есть — «Анализ аудио» берёт вокал, а не падает.

    Кнопка «Анализ аудио» активна после загрузки файла, значит у неё обязан быть
    файл для работы. Раньше активная кнопка упиралась в «Плеер пуст».
    """
    stem = "song"
    wav = tmp_path / f"{stem}.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")
    vocals = tmp_path / f"{stem}_vocals.wav"
    sf.write(str(vocals), np.zeros(1000, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        app.player.stop()
        app.player._current_path = None            # плеер пуст
        app.output_dir = tmp_path
        app._load_file(str(wav))
        app.player._current_path = None            # _load_file снова его заполнил

        src, label = app._analysis_source()
        assert src == vocals
        assert "вокал" in label
    finally:
        app.destroy()


def test_analysis_source_is_player_file_when_it_plays(tmp_path):
    """Если в плеере есть файл — анализируем именно его, без подстановок."""
    wav = tmp_path / "song.wav"
    sf.write(str(wav), np.zeros(44100, dtype=np.float32), 44100, subtype="PCM_16")

    app = _make_window()
    try:
        app.output_dir = tmp_path
        app._load_file(str(wav))

        src, label = app._analysis_source()
        assert src == wav
        assert label == "song.wav"
    finally:
        app.destroy()


def _ancestors(widget):
    """Все родители виджета — от него самого до корня окна."""
    chain = []
    w = widget
    while w is not None:
        chain.append(w)
        w = getattr(w, "master", None)
    return chain


def test_tabs_have_their_own_buttons():
    """У каждой вкладки — свой набор кнопок: девять кнопок сразу больше не висят."""
    app = _make_window()
    try:
        assert app.tab_separate in _ancestors(app.btn_separate)
        assert app.tab_separate in _ancestors(app.btn_cancel)
        assert app.tab_transcribe in _ancestors(app.btn_voc_midi)
        assert app.tab_transcribe in _ancestors(app.btn_inst_midi)
        assert app.tab_transcribe in _ancestors(app.btn_piano)
        assert app.tab_analyze in _ancestors(app.btn_analyze)
        assert app.tab_analyze in _ancestors(app.btn_analyze_midi)
        assert app.tab_analyze in _ancestors(app.btn_super_pair)
        assert app.tab_analyze in _ancestors(app.btn_open_analysis)
    finally:
        app.destroy()


def test_header_has_one_menu_instead_of_two_buttons():
    """«Тема» и «Настройки» свёрнуты в одно меню «⚙», папка — одна кнопка."""
    app = _make_window()
    try:
        header = app.winfo_children()[0]
        texts = [str(w.cget("text")) for w in header.winfo_children()
                 if hasattr(w, "cget")]
        # Кнопка меню в шапке ровно одна — та самая, что открывает меню.
        assert app.btn_menu in header.winfo_children()
        # DECISION: проверяем не символ «⚙», а инвариант «иконка ИЛИ символ».
        # Раньше здесь требовался текст «⚙», и это закрепляло баг: символ
        # рисовался поверх загруженной иконки, в шапке было две шестерёнки
        # одна на другой (QA-прогон 11.10.2026).
        assert (app.btn_menu.cget("text") == ""
                or getattr(app.btn_menu, "_ast_icon", None) is None), \
            "запасной символ рисуется вместе с иконкой"
        assert not [t for t in texts if "Тема" in t], texts
        assert len([t for t in texts if "папку" in t.lower()]) == 1, texts
    finally:
        app.destroy()


def test_menu_has_settings_and_theme_items():
    """Меню «⚙» открывается, а его пункты действительно переключают состояние."""
    app = _make_window()
    try:
        app._open_menu()
        assert app._menu_popup is not None

        app._close_menu()
        assert app._menu_popup is None

        was = app.theme_name
        app._toggle_theme()
        assert app.theme_name != was, "тема не переключилась"
    finally:
        app.destroy()


# ---------- Список движков ----------
def test_registry_has_groups_and_descriptions():
    """У каждого движка есть группа и описание — иначе список снова станет плоским."""
    from core.engines import ENGINES, engines_by_group

    assert len(ENGINES) == 26
    assert len(engines_by_group()) >= 6, "групп меньше шести"

    for spec in ENGINES.values():
        assert spec.group and spec.group != "Прочее", f"{spec.id}: нет группы"
        assert len(spec.description) > 20, f"{spec.id}: описание пустое"
        assert spec.speed_hint, f"{spec.id}: не указана скорость"
        assert spec.tooltip.count("\n") >= 1


def test_registry_is_understandable_without_music_knowledge():
    """
    Главное требование к витрине: название и описание модели должны быть
    понятны человеку, который не знает, кто такие Viperx и Duality.

    Проверяем это механически: у каждой модели есть plain_name и plain_desc,
    а техническое имя (в нём живут фамилии авторов) в понятные поля не попало.
    """
    from core.engines import ENGINES

    for spec in ENGINES.values():
        assert spec.plain_name, f"{spec.id}: нет понятного названия"
        assert len(spec.plain_desc) > 15, f"{spec.id}: нет понятного описания"
        assert spec.plain_desc.endswith((".", "!")), \
            f"{spec.id}: описание должно быть фразой"
        assert spec.technical_name, f"{spec.id}: нет технической подписи"

    # Понятные названия уникальны: два «Быстро и просто» в списке — это
    # неоднозначный выбор.
    names = [s.plain_name for s in ENGINES.values()]
    assert len(names) == len(set(names))


def test_groups_are_tasks_not_authors():
    """Группы — это задачи («Убрать вокал»), а не фамилии авторов моделей."""
    from core.engines import ENGINES, group_caption, group_order

    # DECISION: Demucs в списке нет — это не автор, а название технологии
    # (так же, как «BS» в названии гибридной группы): сказать «Demucs» про
    # модель, которая считает за минуты, понятнее, чем придумывать синоним.
    authors = ("viperx", "duality", "gabox", "becruily", "unwa", "melband")
    for group in group_order():
        assert not any(a in group.lower() for a in authors), \
            f"группа «{group}» названа по автору, а не по задаче"
        assert group_caption(group), f"у группы «{group}» нет пояснения"

    # Каждая группа должна быть непустой: пустой раздел в списке — мусор.
    used = {s.group for s in ENGINES.values()}
    assert set(group_order()) == used


def test_presets_resolve_to_real_engines():
    """Кнопка-пресет обязана вести к существующей модели из нужной группы."""
    from core.engines import ENGINES, PRESETS, preset_engine_id

    assert len(PRESETS) >= 3, "нужны три понятных варианта на выбор"
    for preset in PRESETS:
        assert preset.title and preset.subtitle, "у пресета нет текста"
        engine_id = preset_engine_id(preset)
        assert engine_id in ENGINES, f"пресет «{preset.title}» ведёт в никуда"
        assert ENGINES[engine_id].group == preset.group, \
            f"пресет «{preset.title}» выбирает модель из чужой группы"


def test_group_order_covers_all_engines():
    """Ни одна модель не «потерялась» между группами: сумма по группам = реестр."""
    from core.engines import ENGINES, engines_by_group

    total = sum(len(specs) for specs in engines_by_group().values())
    assert total == len(ENGINES)


def test_engine_selector_shows_presets_first():
    """
    При открытии видно ПРЕСЕТЫ, а не 26 названий моделей.

    Это и есть суть переделки: человек без музыкального образования должен
    выбирать из трёх понятных вариантов, а не из технического списка.
    """
    from core.engines import PRESETS

    app = _make_window()
    try:
        selector = app.engine_combo
        selector.open()

        texts = _all_texts(selector._popup)
        assert "Что вы хотите сделать?" in texts
        for preset in PRESETS:
            assert any(preset.title in t for t in texts), \
                f"пресета «{preset.title}» нет в окне"

        # Полный список спрятан, пока его не попросили.
        assert not selector.is_expanded
        assert "Все модели по задачам" not in texts

        selector.close()
    finally:
        app.destroy()


def test_engine_selector_expands_full_list_on_demand():
    """Кнопка «Показать все модели» раскрывает группы-задачи и строки моделей."""
    from core.engines import ENGINES, engines_by_group

    app = _make_window()
    try:
        selector = app.engine_combo
        assert sorted(selector.values()) == sorted(s.label for s in ENGINES.values())

        selector.open()
        selector._toggle_list()
        assert selector.is_expanded

        texts = _all_texts(selector._popup)
        joined = "\n".join(texts)
        for group in engines_by_group():
            assert group in texts, f"группы «{group}» нет в списке"
        for spec in ENGINES.values():
            # Понятное имя — крупной строкой, рядом пометки (✅ / «рекомендуется»),
            # поэтому проверяем вхождение, а не точное равенство.
            assert spec.display_name in joined, f"нет понятного имени {spec.id}"
            # Техническая подпись живёт в серой строке под названием —
            # внутри скобок, вместе с описанием и скоростью.
            assert spec.label in joined, f"нет технической подписи {spec.id}"

        assert len(selector._tips) == len(ENGINES), "не у всех строк есть подсказка"

        selector._toggle_list()
        assert not selector.is_expanded
        assert selector._tips == [], "подсказки остались после сворачивания"

        selector.close()
        assert not selector.is_open
        assert selector._tips == [], "подсказки остались после закрытия"
    finally:
        app.destroy()


def test_preset_click_selects_engine():
    """Нажатие на пресет выбирает модель, а не просто закрывает окно."""
    from core.engines import ENGINES, PRESETS, preset_engine_id

    app = _make_window()
    try:
        selector = app.engine_combo
        selector.set("demucs:htdemucs")
        selector.open()

        preset = PRESETS[0]
        selector._apply_preset(preset)

        expected = preset_engine_id(preset)
        assert selector.get() == expected
        assert selector.label() == ENGINES[expected].label
        assert selector.display_label() == ENGINES[expected].display_name
        assert not selector.is_open, "после выбора окно должно закрыться"
        # Модель выбрана — можно разделять.
        assert app.btn_separate.cget("state") != "disabled" or not app.ui_state.has_source
    finally:
        app.destroy()


def test_selected_and_recommended_are_marked_in_list():
    """Выбранная модель помечена галочкой, рекомендуемые — словом «рекомендуется»."""
    from core.engines import ENGINES

    app = _make_window()
    try:
        selector = app.engine_combo
        selected = "demucs:htdemucs_ft"
        selector.set(selected)
        selector.open()
        selector._toggle_list()

        texts = _all_texts(selector._popup)
        assert any(t.startswith("✅") and ENGINES[selected].display_name in t
                   for t in texts), "выбранная модель не помечена галочкой"
        assert any("рекомендуется" in t for t in texts), "нет пометки «рекомендуется»"
        selector.close()
    finally:
        app.destroy()


def _all_texts(widget):
    """Все подписи внутри виджета (включая вложенные)."""
    out = []
    try:
        out.append(str(widget.cget("text")))
    except Exception:
        pass
    for child in widget.winfo_children():
        out.extend(_all_texts(child))
    return out


def test_engine_selector_get_set():
    """get() возвращает id движка и понимает set() — как обычный комбобокс."""
    from core.engines import get_engine

    app = _make_window()
    try:
        app.engine_combo.set("demucs:htdemucs")
        assert app.engine_combo.get() == "demucs:htdemucs"
        assert app.engine_combo.label() == get_engine("demucs:htdemucs").label

        # Мусорный id не должен ронять окно
        app.engine_combo.set("такого движка нет")
        assert app.engine_combo.label() == ""
    finally:
        app.destroy()


# ---------- Подсказки ----------
def test_tooltip_shows_and_hides():
    """Подсказка появляется по таймеру и исчезает, когда мышь ушла."""
    from ui.tooltip import ToolTip

    root = ctk.CTk()
    root.withdraw()
    try:
        button = ctk.CTkButton(root, text="кнопка")
        button.pack()
        tip = ToolTip(button, "текст подсказки", delay_ms=1)

        tip._show()
        assert tip._tooltip is not None, "подсказка не появилась"

        tip._on_leave()
        assert tip._tooltip is None, "подсказка не исчезла"
    finally:
        root.destroy()


def test_tooltip_ignores_empty_text():
    """Пустая подсказка не рисует пустое окошко."""
    from ui.tooltip import ToolTip

    root = ctk.CTk()
    root.withdraw()
    try:
        button = ctk.CTkButton(root, text="кнопка")
        button.pack()
        tip = ToolTip(button, "")
        tip._show()
        assert tip._tooltip is None
    finally:
        root.destroy()


def _has_tooltip(widget) -> bool:
    """
    Есть ли на виджете подсказка.

    У кнопок CustomTkinter bind() переопределён и навешивает обработчик на
    внутренний Canvas — спрашивать надо именно у него.
    """
    target = getattr(widget, "_canvas", widget)
    return bool(target.bind("<Enter>"))


def test_buttons_carry_tooltips(tmp_path):
    """
    Пояснения переехали в подсказки: они есть у кнопок транскрипции и анализа,
    а из статус-бара текст про монофонию убран.
    """
    app = _make_window()
    try:
        for name in ("btn_voc_midi", "btn_inst_midi", "btn_piano", "btn_analyze",
                     "btn_analyze_midi", "btn_super_pair", "btn_separate"):
            assert _has_tooltip(getattr(app, name)), f"у {name} нет подсказки"

        text = str(app.status.cget("text"))
        assert "монофонич" not in text.lower()
        assert "torchcrepe" not in text.lower()
    finally:
        app.destroy()
