"""
Гибридное разделение:
  - Вокал  → Demucs (htdemucs_ft)    ← лучший SDR по вокалу
  - Минус  → BS-Roformer-SW          ← лучший SDR по инструменталу
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from core.separator import AudioSeparator
from core.separator_bs import BSAudioSeparator

logger = logging.getLogger(__name__)


class HybridSeparationError(Exception):
    """Ошибка гибридного разделения."""


class HybridSeparator:
    """
    Demucs htdemucs_ft → вокал.
    BS-Roformer-SW    → минус.
    Возвращает единый словарь {vocals, instrumental}.
    """

    def __init__(
        self,
        demucs_model: str = "htdemucs_ft",
        bs_model: str = "BS-Roformer-SW.ckpt",
    ) -> None:
        self.demucs_model = demucs_model
        self.bs_model = bs_model
        logger.info(
            "HybridSeparator init",
            extra={"ctx_demucs": demucs_model, "ctx_bs": bs_model},
        )

    def separate(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        progress_cb=None,
    ) -> dict[str, Path]:
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
            sep_demucs = AudioSeparator(model_name=self.demucs_model)
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
            sep_bs = BSAudioSeparator(model_name=self.bs_model)
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
            progress_cb(0.98, "Сборка результата")

        vocals_final = out_dir / f"{base}_vocals.wav"
        instr_final = out_dir / f"{base}_instrumental.wav"

        try:
            shutil.copy2(demucs_result["vocals"], vocals_final)
            shutil.copy2(bs_result["instrumental"], instr_final)
        except Exception as e:
            raise HybridSeparationError(f"Не удалось собрать финальные файлы: {e}") from e

        if progress_cb:
            progress_cb(1.0, "Готово")

        logger.info(
            "Гибридное разделение завершено",
            extra={"ctx_vocals": str(vocals_final), "ctx_instr": str(instr_final)},
        )
        return {"vocals": vocals_final, "instrumental": instr_final}