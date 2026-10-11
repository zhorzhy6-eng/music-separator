"""
Гибридное разделение (переехало из core/separator_hybrid.py без изменения логики):
  - Вокал  → Demucs (htdemucs_ft)         ← лучший SDR по вокалу
  - Минус  → BS-Roformer / Mel-Band       ← лучший SDR по инструменталу

Уровень финальной пары выставляется ОДИН раз — уже после сборки, общим
коэффициентом на вокал и минус (core.dsp.normalize_pair_files). Ветки свою пару
не нормируют (normalize_pair_output=False), иначе каждая тянула бы уровень на
себя и взаимный баланс вокала и минуса уезжал.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import soundfile as sf

from core.config import get_config
from core.dsp import normalize_pair_files
from core.engines.base import ProgressCb, Separator
from core.engines.demucs_engine import DemucsEngine
from core.paths import MODELS_DIR
from core.stems import StemSet
from core.utils import cleanup_temp_dirs

logger = logging.getLogger(__name__)


class HybridSeparationError(Exception):
    """Ошибка гибридного разделения."""


def _assert_aligned(a_path: Path, b_path: Path, tol_ms: float = 1.0) -> None:
    """
    Проверяет, что вокал и минус выровнены: один sample rate и совпадающие длины.

    Вокал приходит из Demucs (длина = длине входа), минус — из BS-Roformer через
    ffmpeg с принудительным -ar 44100. Совпадение длин никем не гарантировано,
    а рассинхрон вокала и минуса на слух почти не заметен — поэтому проверяем явно.
    """
    info_a = sf.info(str(a_path))
    info_b = sf.info(str(b_path))

    if info_a.samplerate != info_b.samplerate:
        raise HybridSeparationError(
            f"Sample rate не совпадают: {info_a.samplerate} vs {info_b.samplerate}"
        )

    tol_frames = int(tol_ms * info_a.samplerate / 1000)
    if abs(info_a.frames - info_b.frames) > tol_frames:
        raise HybridSeparationError(
            f"Длины не совпадают: {info_a.frames} vs {info_b.frames} кадров "
            f"(разница {abs(info_a.frames - info_b.frames)} кадров, "
            f"допуск {tol_frames})"
        )

    logger.info(
        "Выравнивание проверено",
        extra={"ctx_vocals_frames": info_a.frames, "ctx_instr_frames": info_b.frames,
               "ctx_sample_rate": info_a.samplerate},
    )


class HybridEngine:
    """
    Demucs (вокал) + BS-Roformer (минус). Реализует договор Separator.

    Старое имя класса — HybridSeparator (оставлено ниже как совместимость).
    """

    name = "Hybrid"

    def __init__(
        self,
        demucs_model: str = "htdemucs_ft",
        bs_model: str = "BS-Roformer-SW.ckpt",
        use_api: bool = True,
    ) -> None:
        self.demucs_model = demucs_model
        self.bs_model = bs_model
        # DECISION: по умолчанию BS-ветка гибрида идёт через Python-API
        # (BSEngineAPI), как и одиночный BS-движок. Раньше здесь всегда
        # поднимался BSEngine — запуск audio-separator.exe отдельным процессом:
        # модель грузилась заново на каждый трек (+10 с), прогресс вылавливался
        # из stdout чужого процесса и выглядел как «застыло на 5 %», а в лог
        # сыпались полоски tqdm. use_api=False возвращает прежний путь.
        self._use_api = bool(use_api)
        self._bs_engine = None                # BSEngineAPI | BSEngine | None
        logger.info(
            "HybridEngine init",
            extra={"ctx_demucs": demucs_model, "ctx_bs": bs_model,
                   "ctx_bs_api": self._use_api},
        )

    def _build_bs_engine(self):
        """
        BS-движок для минуса: Python-API или запасной subprocess.

        Настройки берём из конфига теми же getattr с запасным значением, что и
        в фабрике (core/engines/__init__.py): старый settings.json без новых
        полей не должен ломать запуск. Каталог моделей — общий, models/.
        """
        if not self._use_api:
            # Запасной путь: тот же движок через audio-separator.exe
            from core.engines.bs_engine import BSEngine

            return BSEngine(model_name=self.bs_model, normalize_pair_output=False)

        from core.engines.bs_engine_api import BSEngineAPI

        cfg = get_config()
        return BSEngineAPI(
            model_name=self.bs_model,
            model_dir=MODELS_DIR,
            use_fp16=bool(getattr(cfg, "bs_use_fp16", False)),
            batch_size=int(getattr(cfg, "bs_batch_size", 0) or 0),
            keep_loaded=bool(getattr(cfg, "bs_keep_model_loaded", True)),
            normalize_pair_output=False,      # уровень выставит сборщик пары
        )

    def cancel(self) -> bool:
        """
        Прерывает работу: у гибрида долгий шаг — это BS-Roformer (внешний процесс).
        Demucs считается внутри нашего же процесса, его так не прервать.
        """
        return self._bs_engine.cancel() if self._bs_engine is not None else False

    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        progress_cb=None,
    ) -> StemSet:
        in_path = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise HybridSeparationError(f"Файл не найден: {in_path}")

        base = in_path.stem

        # Временные папки под каждый движок
        demucs_dir = out_dir / "_hybrid_demucs"
        bs_dir = out_dir / "_hybrid_bs"
        demucs_dir.mkdir(parents=True, exist_ok=True)
        bs_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Шаг 1: Demucs → вокал ----------
        if progress_cb:
            progress_cb(0.02, f"Demucs {self.demucs_model} (вокал)")
        logger.info("Hybrid: запуск Demucs", extra={"ctx_model": self.demucs_model})

        try:
            sep_demucs = DemucsEngine(model_name=self.demucs_model,
                                      normalize_pair_output=False)
            demucs_result = sep_demucs.separate(
                in_path, demucs_dir,
                progress_cb=(
                    (lambda p, s: progress_cb(0.02 + p * 0.48, f"Demucs: {s}"))
                    if progress_cb else None
                ),
            )
        except Exception as e:
            logger.exception("Hybrid: Demucs упал")
            raise HybridSeparationError(f"Demucs не справился: {e}") from e

        # ---------- Шаг 2: BS-Roformer → минус ----------
        if progress_cb:
            progress_cb(0.52, f"BS-Roformer {self.bs_model} (минус)")
        logger.info("Hybrid: запуск BS-Roformer", extra={"ctx_model": self.bs_model})

        try:
            sep_bs = self._build_bs_engine()
            self._bs_engine = sep_bs          # чтобы гибрид можно было отменить
            bs_result = sep_bs.separate(
                in_path, bs_dir,
                progress_cb=(
                    (lambda p, s: progress_cb(0.52 + p * 0.46, f"BS-Roformer: {s}"))
                    if progress_cb else None
                ),
            )
        except Exception as e:
            logger.exception("Hybrid: BS-Roformer упал")
            raise HybridSeparationError(f"BS-Roformer не справился: {e}") from e

        # ---------- Шаг 3: Собираем финальные файлы ----------
        if progress_cb:
            progress_cb(0.96, "Сборка результата")

        vocals_final = out_dir / f"{base}_vocals.wav"
        instr_final = out_dir / f"{base}_instrumental.wav"

        # Рассинхрон вокала и минуса на слух почти не заметен, поэтому
        # проверяем выравнивание ДО копирования финальных файлов
        _assert_aligned(demucs_result["vocals"], bs_result["instrumental"])

        try:
            shutil.copy2(demucs_result["vocals"], vocals_final)
            shutil.copy2(bs_result["instrumental"], instr_final)
        except Exception as e:
            raise HybridSeparationError(f"Не удалось собрать финальные файлы: {e}") from e

        # ---------- Шаг 4: ОДИН общий коэффициент на финальную пару ----------
        # Вокал пришёл из Demucs, минус — из BS-Roformer. Ветки свои пары не
        # нормировали, поэтому взаимный уровень выставляется здесь и ровно один раз.
        if progress_cb:
            progress_cb(0.98, "Нормализация пары")
        gain_db = normalize_pair_files(vocals_final, instr_final)
        if gain_db is None:
            logger.warning("Финальная нормализация пары не выполнена — проверь стемы")
        else:
            logger.info("Финальная нормализация пары: коэффициент %.2f dB", gain_db)

        # ---------- Шаг 5: уборка рабочих папок ----------
        # Каждый прогон гибрида оставлял в output/ ~163 МБ (Demucs-стемы,
        # BS-стемы и промежуточные FLAC-и). Теперь убираем за собой.
        if get_config().keep_temp_dirs:
            logger.info("keep_temp_dirs=True — рабочие папки %s и %s оставлены",
                        demucs_dir.name, bs_dir.name)
        else:
            cleanup_temp_dirs([demucs_dir, bs_dir])

        if progress_cb:
            progress_cb(1.0, "Готово")

        logger.info(
            "Гибридное разделение завершено",
            extra={"ctx_vocals": str(vocals_final), "ctx_instr": str(instr_final)},
        )
        return StemSet(vocals=vocals_final, instrumental=instr_final)


# ---------- Совместимость со старыми именами ----------
# Раньше модуль назывался core/separator_hybrid.py, а класс — HybridSeparator.
HybridSeparator = HybridEngine

__all__ = ["HybridEngine", "HybridSeparator", "HybridSeparationError",
           "_assert_aligned"]