"""
Реестр движков («меню»). Единственный источник правды о моделях.

Добавить модель = добавить одну строку в ENGINES.
UI строится из ENGINES, валидация — из ENGINES, фабрика make_separator — из ENGINES.

Почему так: раньше список моделей лежал в ТРЁХ местах (ui/app_window.py,
core/separator.py, core/separator_bs.py) и они успели разъехаться — часть
моделей была доступна из кода, но недоступна из интерфейса, и наоборот.
Теперь UI — просто витрина этого списка.

Термины:
  kind — семейство движка: "demucs" (локальная модель Demucs),
         "bs" (BS-Roformer / Mel-Band Roformer),
         "hybrid" (вокал от Demucs + минус от BS).

Что внутри пакета:
  base.py           — договор Separator (что обязан уметь любой движок);
  demucs_engine.py  — Demucs;
  bs_engine_api.py  — BS-Roformer через Python-API audio_separator (по умолчанию);
  bs_engine.py      — BS-Roformer через audio-separator.exe (запасной путь);
  hybrid_engine.py  — вокал от Demucs + минус от BS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.engines.base import EngineCancelled, ProgressCb, Separator

# DECISION: id строится как "<kind>:<model>". Он не зависит от подписи в UI,
# поэтому переименование кнопки в интерфейсе не ломает настройки и CLI.
# В settings.json при этом остаётся и старая подпись (label) — см. resolve_engine_id.

__all__ = [
    "EngineSpec", "ENGINES", "DEFAULT_ENGINE_ID",
    "get_engine", "engines_by_label", "all_labels", "resolve_engine_id",
    "engines_by_group", "group_order", "group_caption",
    "PRESETS", "Preset", "preset_engine_id",
    "make_separator", "Separator", "ProgressCb", "EngineCancelled",
]

# --- Группы движков для витрины UI -------------------------------
# DECISION: раньше группировка была ПО АВТОРУ модели (Viperx, Duality, Gabox,
# Becruily) — это удобно тому, кто читает HuggingFace, и бесполезно всем
# остальным: «Viperx-1297» и «Gabox bleedless v2» ничего не говорят о том, что
# получится на выходе. Теперь группа — это ЗАДАЧА, которую решает модель.
# Порядок групп задаётся одним списком здесь, чтобы переставить разделы не
# трогая ни один EngineSpec.
GROUP_KARAOKE = "🎤 Убрать вокал (для караоке)"
GROUP_INSTRUMENTAL = "🎸 Минус как в оригинале"
GROUP_VOCALS = "🎙 Только вокал"
GROUP_FAST = "⚡ Быстро и просто"
GROUP_HYBRID = "🎹 Гибриды: Demucs + BS"
GROUP_OTHER = "🔧 Прочее"

GROUP_ORDER: tuple[str, ...] = (
    GROUP_KARAOKE,
    GROUP_INSTRUMENTAL,
    GROUP_VOCALS,
    GROUP_FAST,
    GROUP_HYBRID,
    GROUP_OTHER,
)

# Пояснение к группе: что человек получит, выбрав любую модель из неё.
GROUP_CAPTIONS: dict[str, str] = {
    GROUP_KARAOKE: "Голос уходит полностью, музыка остаётся. Для пения.",
    GROUP_INSTRUMENTAL: "Музыка сохраняется максимально полно. Может "
                        "остаться лёгкий след голоса.",
    GROUP_VOCALS: "Нужен отдельно голос — для акапеллы, сэмплов, разбора партии.",
    GROUP_FAST: "Считает за минуты и работает почти на любой видеокарте. "
                "Качество ниже, чем у BS-моделей.",
    GROUP_HYBRID: "Голос берёт Demucs, музыку — BS-модель. Дольше всех, "
                  "результат ровнее.",
    GROUP_OTHER: "Редкие и старые модели: для сравнения и особых случаев.",
}


@dataclass(frozen=True)
class EngineSpec:
    """Описание одного движка: что показать в UI и что запустить внутри."""

    id: str
    label: str
    kind: Literal["demucs", "bs", "hybrid"]
    model: str
    companion: str | None = None          # для hybrid — модель Demucs (вокал)
    stems: tuple[str, ...] = ("vocals", "instrumental")
    # DECISION: True — BS-модель считается Python-API audio_separator (модель
    # грузится один раз за сессию), False — прежний запуск audio-separator.exe.
    # Переключатель оставлен на случай проблем с API: одна строка в реестре.
    use_api: bool = True

    # --- Поля для витрины UI (на работу движка не влияют) ---
    # DECISION: подсказки (группа, описание, SDR, VRAM, скорость) живут в реестре,
    # а не в интерфейсе. 26 моделей — это данные о моделях, а не о кнопках:
    # добавление модели по-прежнему = одна строка в ENGINES.
    group: str = GROUP_OTHER             # группа-ЗАДАЧА в списке движков
    description: str = ""                 # текст всплывающей подсказки
    sdr_hint: float | None = None         # ожидаемый SDR (качество разделения)
    vram_hint_mb: int | None = None       # сколько нужно видеопамяти
    speed_hint: str = ""                  # «быстро» / «средне» / «медленно»

    # --- Поля «человеческим языком» ---
    # DECISION: label остаётся техническим именем (по нему узнают модель на
    # форумах, и оно лежит в settings.json), а для витрины добавлены отдельные
    # поля. Так мы не ломаем старые настройки и при этом можем писать в списке
    # «Максимум инструментала» вместо «Viperx-1297».
    plain_name: str = ""                  # понятное название для не-музыканта
    plain_desc: str = ""                  # понятное описание в одну строку
    technical_name: str = ""              # «Viperx-1297, SDR 12.98» — в подсказку
    recommended: bool = False             # пометка «рекомендуется» в списке

    @property
    def display_name(self) -> str:
        """
        Что показать крупно в списке моделей.

        Пустое plain_name — не ошибка: новая модель без описания должна
        появиться в интерфейсе, а не пропасть. Тогда показываем label.
        """
        return self.plain_name or self.label

    @property
    def technical_line(self) -> str:
        """Строка «для профи»: техническое имя модели, иначе — её label."""
        return self.technical_name or self.label

    @property
    def name(self) -> str:
        """Имя движка для логов и CLI."""
        return self.model

    @property
    def tooltip(self) -> str:
        """
        Готовая подсказка для наведения мыши: техническое имя, пояснение, цифры.

        Собирается здесь, а не в UI: числа (SDR, VRAM, скорость) — свойство
        модели, и форматировать их должен тот, кто ими владеет.

        Важно про повторы: plain_desc и description говорят об одном и том же
        разными словами — первое для витрины, второе подробнее. Чтобы в
        подсказке не было двух абзацев про одно и то же, подробное описание
        добавляется только тогда, когда оно НЕ начинается с понятного.
        """
        lines = [self.technical_line]
        if self.plain_desc:
            lines.append(self.plain_desc)
        full = self.description.strip()
        plain = self.plain_desc.strip()
        if full and full != plain and not (plain and full.startswith(plain)):
            lines.append(full)
        facts: list[str] = []
        if self.sdr_hint is not None:
            facts.append(f"SDR ≈ {self.sdr_hint:.2f}")
        if self.speed_hint:
            facts.append(f"скорость: {self.speed_hint}")
        if self.vram_hint_mb:
            gb = self.vram_hint_mb / 1024
            facts.append(f"видеопамять: ~{gb:.1f} ГБ")
        if facts:
            lines.append(" • ".join(facts))
        return "\n".join(lines)


ENGINES: dict[str, EngineSpec] = {
    # ============================================================
    #  ⚡ Быстро и просто — Demucs (локальная модель, без BS)
    # ============================================================
    "demucs:htdemucs": EngineSpec(
        id="demucs:htdemucs",
        label="Demucs: htdemucs (база)",
        kind="demucs",
        model="htdemucs",
        group=GROUP_FAST,
        description="Базовый Demucs. Качество скромнее, чем у BS-Roformer, "
                    "зато считается на любой видеокарте и почти не ест память.",
        plain_name="Самый быстрый",
        plain_desc="Разделяет за 1–2 минуты. Качество скромнее, но видеопамяти "
                   "почти не нужно.",
        technical_name="Demucs htdemucs, SDR 8.1",
        sdr_hint=8.1, speed_hint="быстро", vram_hint_mb=1536,
    ),
    "demucs:htdemucs_ft": EngineSpec(
        id="demucs:htdemucs_ft",
        label="Demucs: htdemucs_ft ✅ +SDR",
        kind="demucs",
        model="htdemucs_ft",
        group=GROUP_FAST,
        description="Тот же Demucs, но дообученный (fine-tuned): чище вокал. "
                    "Считается в 4 раза дольше базы — это лучший вариант, "
                    "если видеопамяти мало.",
        plain_name="Быстро и просто",
        plain_desc="Разделяет за 3–5 минут. Хорошее качество и работает "
                   "почти на любой видеокарте.",
        technical_name="Demucs htdemucs_ft, SDR 8.6",
        recommended=True,
        sdr_hint=8.6, speed_hint="средне", vram_hint_mb=1536,
    ),

    # ============================================================
    #  🎤 Убрать вокал (для караоке) — минус без следов голоса
    # ============================================================
    "bs:mel_band_roformer_karaoke_becruily.ckpt": EngineSpec(
        id="bs:mel_band_roformer_karaoke_becruily.ckpt",
        label="BS-Roformer: Becruily 🎤 karaoke",
        kind="bs",
        model="mel_band_roformer_karaoke_becruily.ckpt",
        group=GROUP_KARAOKE,
        description="Для караоке — чистое отделение вокала от минуса. "
                    "Минус специально обучен не «тянуть» за собой голос.",
        plain_name="Караоке",
        plain_desc="Чистое отделение вокала. Для пения — самый понятный выбор.",
        technical_name="Becruily karaoke (mel_band_roformer_karaoke_becruily)",
        recommended=True,
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt": EngineSpec(
        id="bs:mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
        label="BS-Roformer: Gabox 🎸 bleedless v2",
        kind="bs",
        model="mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
        group=GROUP_KARAOKE,
        description="Минус без протечек вокала. Для караоке: голос уходит, "
                    "музыка остаётся. Лучший баланс в линейке bleedless.",
        plain_name="Без протечек вокала",
        plain_desc="В минусе не остаётся следов голоса. Для караоке и минусовок.",
        technical_name="Gabox bleedless v2",
        recommended=True,
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt": EngineSpec(
        id="bs:mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
        label="BS-Roformer: Gabox 🎸 bleedless v3",
        kind="bs",
        model="mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
        group=GROUP_KARAOKE,
        description="Минус без протечек вокала. Третья версия: чище всего "
                    "убирает голос, но сильнее всего вмешивается в музыку.",
        plain_name="Голос убран начисто",
        plain_desc="Самый чистый минус, но музыка звучит чуть проще.",
        technical_name="Gabox bleedless v3",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt": EngineSpec(
        id="bs:mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt",
        label="BS-Roformer: Gabox 🎸 bleedless v1",
        kind="bs",
        model="mel_band_roformer_instrumental_bleedless_v1_gabox.ckpt",
        group=GROUP_KARAOKE,
        description="Минус без протечек вокала. Первая версия — самая "
                    "осторожная: музыку может слегка «подсушивать».",
        plain_name="Без протечек вокала (старая версия)",
        plain_desc="То же, что v2, но бережнее к музыке и чуть больше следов "
                   "голоса.",
        technical_name="Gabox bleedless v1",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_karaoke_gabox_v2.ckpt": EngineSpec(
        id="bs:mel_band_roformer_karaoke_gabox_v2.ckpt",
        label="BS-Roformer: Karaoke-Gabox-V2",
        kind="bs",
        model="mel_band_roformer_karaoke_gabox_v2.ckpt",
        group=GROUP_KARAOKE,
        description="Караоке от Gabox, вторая версия: музыка плотнее, вокал "
                    "убирается так же чисто. Лучший выбор среди караоке.",
        plain_name="Караоке с плотной музыкой",
        plain_desc="Голос уходит, инструменты звучат полнее. Для пения.",
        technical_name="Karaoke-Gabox-V2",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_karaoke_gabox.ckpt": EngineSpec(
        id="bs:mel_band_roformer_karaoke_gabox.ckpt",
        label="BS-Roformer: Karaoke-Gabox",
        kind="bs",
        model="mel_band_roformer_karaoke_gabox.ckpt",
        group=GROUP_KARAOKE,
        description="Караоке от Gabox: минус без голоса, но музыка может "
                    "звучать чуть «жиже». Первая версия.",
        plain_name="Караоке (первая версия)",
        plain_desc="Голос убирает хорошо, музыка звучит чуть беднее.",
        technical_name="Karaoke-Gabox",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": EngineSpec(
        id="bs:mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        label="BS-Roformer: Karaoke-Aufr33",
        kind="bs",
        model="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        group=GROUP_KARAOKE,
        description="Караоке-модель от Aufr33 на базе Viperx. Старше, чем Gabox, "
                    "но всё ещё рабочая.",
        plain_name="Караоке (старая модель)",
        plain_desc="Убирает голос, но грубее новых моделей. Оставлена для "
                   "сравнения.",
        technical_name="Karaoke-Aufr33 Viperx, SDR 10.20",
        sdr_hint=10.2, speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:melband_roformer_instvox_duality_v2.ckpt": EngineSpec(
        id="bs:melband_roformer_instvox_duality_v2.ckpt",
        label="BS-Roformer: Duality-V2 ✅ баланс",
        kind="bs",
        model="melband_roformer_instvox_duality_v2.ckpt",
        group=GROUP_KARAOKE,
        description="Сбалансированный: хорош и для вокала, и для минуса. "
                    "Универсальный выбор, если не хочется думать. "
                    "Стоит по умолчанию.",
        plain_name="Сбалансированный (рекомендуется)",
        plain_desc="Хорош и для вокала, и для минуса. Универсальный выбор.",
        technical_name="Duality-V2 (melband_roformer_instvox_duality_v2), SDR ≈ 12",
        recommended=True,
        sdr_hint=12.0, speed_hint="средне", vram_hint_mb=4096,
    ),

    # ============================================================
    #  🎸 Минус как в оригинале — музыка сохраняется максимально
    # ============================================================
    "bs:model_bs_roformer_ep_317_sdr_12.9755.ckpt": EngineSpec(
        id="bs:model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        label="BS-Roformer: Viperx-1297 🏆 инстр",
        kind="bs",
        model="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        group=GROUP_INSTRUMENTAL,
        description="Самый высокий SDR среди BS-моделей: музыка остаётся "
                    "максимально близкой к оригиналу.",
        plain_name="Максимум инструментала",
        plain_desc="Лучший результат для минуса. Медленно: ~3 минуты на трек "
                   "4 минуты.",
        technical_name="Viperx-1297 (model_bs_roformer_ep_317), SDR 12.98",
        recommended=True,
        sdr_hint=12.98, speed_hint="медленно", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_instrumental_fullness_v3_gabox.ckpt": EngineSpec(
        id="bs:mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
        label="BS-Roformer: Gabox 🎸 fullness v3",
        kind="bs",
        model="mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
        group=GROUP_INSTRUMENTAL,
        description="Максимальная детализация минуса, третья версия: музыки "
                    "больше, чем в v1, следов вокала меньше.",
        plain_name="Максимум деталей",
        plain_desc="Сохраняет все инструменты. Может содержать следы вокала.",
        technical_name="Gabox fullness v3",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_instrumental_fullness_v1_gabox.ckpt": EngineSpec(
        id="bs:mel_band_roformer_instrumental_fullness_v1_gabox.ckpt",
        label="BS-Roformer: Gabox 🎸 fullness v1",
        kind="bs",
        model="mel_band_roformer_instrumental_fullness_v1_gabox.ckpt",
        group=GROUP_INSTRUMENTAL,
        description="Максимальная детализация минуса. Может содержать следы "
                    "вокала — это плата за «полноту» звучания.",
        plain_name="Максимум деталей (старая версия)",
        plain_desc="Много инструментов, но следов голоса больше, чем в v3.",
        technical_name="Gabox fullness v1",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:melband_roformer_inst_v2.ckpt": EngineSpec(
        id="bs:melband_roformer_inst_v2.ckpt",
        label="BS-Roformer: MelBand-Inst-V2",
        kind="bs",
        model="melband_roformer_inst_v2.ckpt",
        group=GROUP_INSTRUMENTAL,
        description="Mel-Band версия, заточенная под минус. Хороший компромисс "
                    "между качеством и временем счёта.",
        plain_name="Минус за разумное время",
        plain_desc="Полный минус без долгого ожидания. Средний вариант.",
        technical_name="MelBand-Inst-V2 (melband_roformer_inst_v2)",
        sdr_hint=12.0, speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:melband_roformer_big_beta5e.ckpt": EngineSpec(
        id="bs:melband_roformer_big_beta5e.ckpt",
        label="BS-Roformer: Big Beta 5e",
        kind="bs",
        model="melband_roformer_big_beta5e.ckpt",
        group=GROUP_INSTRUMENTAL,
        description="Максимум качества, очень медленно. Нужно ~6 ГБ видеопамяти. "
                    "Для одного-двух важных треков, не для потока.",
        plain_name="Максимум качества (очень медленно)",
        plain_desc="Лучший результат, но ждать долго. Нужно много видеопамяти.",
        technical_name="Big Beta 5e (melband_roformer_big_beta5e), SDR ~12",
        sdr_hint=12.0, speed_hint="очень медленно", vram_hint_mb=6144,
    ),
    "bs:melband_roformer_big_beta4.ckpt": EngineSpec(
        id="bs:melband_roformer_big_beta4.ckpt",
        label="BS-Roformer: Big Beta 4",
        kind="bs",
        model="melband_roformer_big_beta4.ckpt",
        group=GROUP_INSTRUMENTAL,
        description="Большая модель: заметно качественнее обычных, но считает "
                    "долго и требует много видеопамяти.",
        plain_name="Очень качественно (старая версия)",
        plain_desc="Почти как Beta 5e, но чуть слабее. Тоже нужна большая "
                   "видеопамять.",
        technical_name="Big Beta 4 (melband_roformer_big_beta4)",
        speed_hint="медленно", vram_hint_mb=6144,
    ),

    # ============================================================
    #  🎙 Только вокал — голос отдельно
    # ============================================================
    "bs:model_bs_roformer_ep_368_sdr_12.9628.ckpt": EngineSpec(
        id="bs:model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        label="BS-Roformer: Viperx-1296 🏆 вокал",
        kind="bs",
        model="model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        group=GROUP_VOCALS,
        description="Лучший для вокала: почти тот же SDR, что у 1297, но вокал "
                    "чище. Берите, когда важнее вокал, а не минус.",
        plain_name="Лучший вокал",
        plain_desc="Голос чище всего. Минус может получиться с протечками.",
        technical_name="Viperx-1296 (model_bs_roformer_ep_368), SDR 12.96",
        recommended=True,
        sdr_hint=12.96, speed_hint="медленно", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_vocals_becruily.ckpt": EngineSpec(
        id="bs:mel_band_roformer_vocals_becruily.ckpt",
        label="BS-Roformer: Becruily 🎤 vocals",
        kind="bs",
        model="mel_band_roformer_vocals_becruily.ckpt",
        group=GROUP_VOCALS,
        description="Заточена под вокал: голос вытаскивает чище, минус может "
                    "быть с протечками. Для выделения акапеллы.",
        plain_name="Чистый голос",
        plain_desc="Голос отделяется аккуратно. Для акапеллы и сэмплов.",
        technical_name="Becruily vocals (mel_band_roformer_vocals_becruily)",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_kim_ft_unwa.ckpt": EngineSpec(
        id="bs:mel_band_roformer_kim_ft_unwa.ckpt",
        label="BS-Roformer: Unwa FT",
        kind="bs",
        model="mel_band_roformer_kim_ft_unwa.ckpt",
        group=GROUP_VOCALS,
        description="Модель Unwa на базе Kim FT: ровный результат на разной "
                    "музыке, без ярких провалов.",
        plain_name="Ровный голос на любой музыке",
        plain_desc="Без провалов на незнакомых жанрах. Надёжный средний вариант.",
        technical_name="Unwa FT (mel_band_roformer_kim_ft_unwa)",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:mel_band_roformer_kim_ft2_bleedless_unwa.ckpt": EngineSpec(
        id="bs:mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
        label="BS-Roformer: Unwa FT2 bleedless",
        kind="bs",
        model="mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
        group=GROUP_VOCALS,
        description="Вторая версия Unwa без протечек: голос в минусе не "
                    "остаётся. Альтернатива Gabox bleedless.",
        plain_name="Ровный голос без следов в минусе",
        plain_desc="Голос чистый, и в минусе его почти не слышно.",
        technical_name="Unwa FT2 bleedless (mel_band_roformer_kim_ft2_bleedless_unwa)",
        speed_hint="средне", vram_hint_mb=4096,
    ),

    # ============================================================
    #  🎹 Гибриды: вокал от Demucs, минус от BS-Roformer
    # ============================================================
    "hybrid:htdemucs_ft+melband_roformer_instvox_duality_v2.ckpt": EngineSpec(
        id="hybrid:htdemucs_ft+melband_roformer_instvox_duality_v2.ckpt",
        label="Hybrid: Demucs_ft + Duality-V2 ✅ баланс",
        kind="hybrid",
        model="melband_roformer_instvox_duality_v2.ckpt",
        companion="htdemucs_ft",
        group=GROUP_HYBRID,
        description="Вокал от Demucs_ft, минус от Duality-V2. Ровный вариант "
                    "«на каждый день».",
        plain_name="Ровный результат на каждый день",
        plain_desc="Голос берёт Demucs, музыку — Duality-V2. Долго, но без "
                   "сюрпризов.",
        technical_name="Demucs_ft (вокал) + Duality-V2 (минус)",
        recommended=True,
        sdr_hint=12.0, speed_hint="очень медленно", vram_hint_mb=4096,
    ),
    "hybrid:htdemucs_ft+model_bs_roformer_ep_317_sdr_12.9755.ckpt": EngineSpec(
        id="hybrid:htdemucs_ft+model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        label="Hybrid: Demucs_ft + Viperx-1297 🏆 инстр",
        kind="hybrid",
        model="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        companion="htdemucs_ft",
        group=GROUP_HYBRID,
        description="Вокал берёт Demucs_ft, минус — лучшая модель для "
                    "инструментала (Viperx-1297). Два движка за один запуск, "
                    "поэтому дольше обычного.",
        plain_name="Лучший минус + надёжный вокал",
        plain_desc="Минус от лучшей модели для инструментала. Дольше всех.",
        technical_name="Demucs_ft (вокал) + Viperx-1297 (минус), SDR 12.98",
        sdr_hint=12.98, speed_hint="очень медленно", vram_hint_mb=4096,
    ),
    "hybrid:htdemucs_ft+mel_band_roformer_karaoke_becruily.ckpt": EngineSpec(
        id="hybrid:htdemucs_ft+mel_band_roformer_karaoke_becruily.ckpt",
        label="Hybrid: Demucs_ft + Becruily 🎤 karaoke",
        kind="hybrid",
        model="mel_band_roformer_karaoke_becruily.ckpt",
        companion="htdemucs_ft",
        group=GROUP_HYBRID,
        description="Вокал от Demucs_ft, минус от караоке-модели Becruily. "
                    "Для караоке-версий.",
        plain_name="Караоке + надёжный вокал",
        plain_desc="Для пения, когда нужен и чистый голос, и чистый минус.",
        technical_name="Demucs_ft (вокал) + Becruily karaoke (минус)",
        speed_hint="очень медленно", vram_hint_mb=4096,
    ),

    # ============================================================
    #  🔧 Прочее — редкие и старые модели
    # ============================================================
    "bs:mel_band_roformer_instrumental_becruily.ckpt": EngineSpec(
        id="bs:mel_band_roformer_instrumental_becruily.ckpt",
        label="BS-Roformer: Becruily 🎸 inst",
        kind="bs",
        model="mel_band_roformer_instrumental_becruily.ckpt",
        group=GROUP_OTHER,
        description="Заточена под минус: музыка остаётся целой, вокал убирается "
                    "агрессивнее. Для минусовок и подпевок.",
        plain_name="Минус от Becruily",
        plain_desc="Музыка целая, голос убирается смелее. Родственник "
                   "караоке-модели.",
        technical_name="Becruily instrumental (mel_band_roformer_instrumental_becruily)",
        speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:melband_roformer_instvoc_duality_v1.ckpt": EngineSpec(
        id="bs:melband_roformer_instvoc_duality_v1.ckpt",
        label="BS-Roformer: Duality-V1 (баланс)",
        kind="bs",
        model="melband_roformer_instvoc_duality_v1.ckpt",
        group=GROUP_OTHER,
        description="Первая «двойная» модель: одинаково ровно работает и с "
                    "вокалом, и с минусом. Предыдущая версия Duality-V2.",
        plain_name="Сбалансированный (старая версия)",
        plain_desc="То же назначение, что у Duality-V2, но результат слабее.",
        technical_name="Duality-V1 (melbandroformer_instvoc_duality_v1), SDR 11.5",
        sdr_hint=11.5, speed_hint="средне", vram_hint_mb=4096,
    ),
    "bs:BS-Roformer-SW.ckpt": EngineSpec(
        id="bs:BS-Roformer-SW.ckpt",
        label="BS-Roformer: SW (базовая)",
        kind="bs",
        model="BS-Roformer-SW.ckpt",
        group=GROUP_OTHER,
        description="Ранняя базовая BS-Roformer. Оставлена для сравнения: "
                    "остальные BS-модели её заметно обходят.",
        plain_name="Самая первая BS-модель",
        plain_desc="Уступает всем остальным BS-моделям. Только для сравнения.",
        technical_name="BS-Roformer-SW, SDR 10.0",
        sdr_hint=10.0, speed_hint="средне", vram_hint_mb=4096,
    ),
}

# Подпись по умолчанию, если настройка default_engine указывает в никуда
DEFAULT_ENGINE_ID = "bs:melband_roformer_instvox_duality_v2.ckpt"


def get_engine(engine_id: str) -> EngineSpec:
    """Движок по id. Неизвестный id — ошибка (молча подставлять другой нельзя)."""
    if engine_id not in ENGINES:
        raise KeyError(f"Неизвестный движок: {engine_id}")
    return ENGINES[engine_id]


def engines_by_label() -> dict[str, str]:
    """{label: id} для построения UI."""
    return {spec.label: spec.id for spec in ENGINES.values()}


def all_labels() -> list[str]:
    """Все подписи в порядке реестра — для выпадающего списка."""
    return [spec.label for spec in ENGINES.values()]


def resolve_engine_id(value: str) -> str:
    """
    Приводит значение из настроек/CLI к id движка.

    Понимает и id ("bs:...ckpt"), и подпись из UI ("BS-Roformer: Duality-V2 ✅ баланс").
    Нужно для обратной совместимости: в config/settings.json и в старых записях
    лежит именно подпись.
    """
    if value in ENGINES:
        return value
    by_label = engines_by_label()
    if value in by_label:
        return by_label[value]
    raise KeyError(f"Неизвестный движок: {value}")


def group_order() -> list[str]:
    """
    Названия групп в том порядке, в каком их показывать.

    Сначала идут группы из GROUP_ORDER, затем — всё, что в реестре есть, а в
    списке порядка забыли (чтобы новая модель не потерялась из интерфейса).
    """
    known = [g for g in GROUP_ORDER if any(s.group == g for s in ENGINES.values())]
    extra = [g for g in dict.fromkeys(s.group for s in ENGINES.values())
             if g not in known]
    return known + extra


def group_caption(group: str) -> str:
    """Пояснение к группе: что человек получит, выбрав любую модель из неё."""
    return GROUP_CAPTIONS.get(group, "")


# ============================================================
#  Пресеты: три больших вопроса вместо 26 моделей
# ============================================================
@dataclass(frozen=True)
class Preset:
    """
    Крупная кнопка над списком моделей: «что вы хотите сделать?».

    Почему отдельная сущность, а не «просто кнопка в UI»: набор пресетов —
    это правило выбора модели (какая модель решает какую задачу), то есть
    данные реестра. Интерфейс лишь рисует то, что здесь написано.
    """

    title: str                            # крупная строка на кнопке
    subtitle: str                         # мелкая строка-пояснение
    group: str                            # какую группу раскрыть
    candidates: tuple[str, ...]           # id моделей по убыванию предпочтения
    icon: str = ""                        # эмодзи для кнопки


# DECISION: в candidates лежит НЕСКОЛЬКО id, а выбирается первый, который есть
# в реестре. Если модель когда-нибудь уберут (или переименуют файл), кнопка
# пресета не сломается — просто возьмёт следующую по качеству. Пустой список
# кандидатов означает «просто раскрыть группу, ничего не менять».
PRESETS: tuple[Preset, ...] = (
    Preset(
        title="Убрать вокал (для караоке)",
        subtitle="Быстро, чисто, без протечек",
        group=GROUP_KARAOKE,
        candidates=("bs:mel_band_roformer_karaoke_becruily.ckpt",
                    "bs:mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt"),
        icon="🎤",
    ),
    Preset(
        title="Получить минус как в оригинале",
        subtitle="Максимальное качество, можно подождать",
        group=GROUP_INSTRUMENTAL,
        candidates=("bs:model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                    "bs:melband_roformer_big_beta5e.ckpt"),
        icon="🎸",
    ),
    Preset(
        title="Быстро и просто",
        subtitle="Demucs, ~3 минуты",
        group=GROUP_FAST,
        candidates=("demucs:htdemucs_ft", "demucs:htdemucs"),
        icon="⚡",
    ),
)


def preset_engine_id(preset: Preset) -> str | None:
    """
    Какую модель выбрать при нажатии на пресет.

    Первый существующий кандидат; None — если у пресета кандидатов нет или
    ни одного из них больше нет в реестре.
    """
    for engine_id in preset.candidates:
        if engine_id in ENGINES:
            return engine_id
    return None


def engines_by_group() -> dict[str, list[EngineSpec]]:
    """
    {группа: [движки]} — для списка с подзаголовками.

    Порядок движков внутри группы — как в реестре ENGINES.
    """
    grouped: dict[str, list[EngineSpec]] = {g: [] for g in group_order()}
    for spec in ENGINES.values():
        grouped.setdefault(spec.group, []).append(spec)
    return {g: specs for g, specs in grouped.items() if specs}


def make_separator(spec: EngineSpec):
    """
    Фабрика: описание движка → готовый сепаратор.

    Импорты внутри функции: demucs и BS тянут тяжёлые библиотеки (torch, onnx),
    а реестр нужен интерфейсу сразу при старте — грузить их ради списка моделей
    незачем.
    """
    if spec.kind == "demucs":
        from core.engines.demucs_engine import DemucsEngine
        return DemucsEngine(model_name=spec.model)
    if spec.kind == "bs":
        if spec.use_api:
            # BS-Roformer через Python-API: модель грузится один раз за сессию.
            from core.config import get_config
            from core.engines.bs_engine_api import BSEngineAPI
            from core.paths import MODELS_DIR

            cfg = get_config()
            # getattr с запасным значением: старый settings.json без новых
            # полей не должен ломать запуск.
            return BSEngineAPI(
                model_name=spec.model,
                model_dir=MODELS_DIR,
                use_fp16=bool(getattr(cfg, "bs_use_fp16", False)),
                batch_size=int(getattr(cfg, "bs_batch_size", 0) or 0),
                keep_loaded=bool(getattr(cfg, "bs_keep_model_loaded", True)),
            )
        from core.engines.bs_engine import BSEngine
        return BSEngine(model_name=spec.model)
    if spec.kind == "hybrid":
        from core.engines.hybrid_engine import HybridEngine
        # use_api — тот же переключатель, что и у одиночного BS: гибрид считает
        # минус тем же BS-движком, поэтому запасной путь (audio-separator.exe)
        # должен остаться доступен и здесь.
        return HybridEngine(demucs_model=spec.companion, bs_model=spec.model,
                            use_api=spec.use_api)
    raise ValueError(f"Неизвестный kind: {spec.kind}")
