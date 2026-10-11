"""
Воркер, который выполняет тяжёлые задачи в отдельном потоке.
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from core.separator import AudioSeparator
from core.transcriber import MidiTranscriber
from core.analyzer import analyze_audio
from core.piano_transcriber import PianoTranscriber
from core.midi_analyzer import analyze_midi

logger = logging.getLogger(__name__)


class TaskType(Enum):
    SEPARATE = auto()
    SEPARATE_BS = auto()
    SEPARATE_HYBRID = auto()
    TRANSCRIBE = auto()
    ANALYZE = auto()
    ANALYZE_PAIR = auto()
    PIANO = auto()
    MIDI_ANALYZE = auto()


@dataclass
class ProgressEvent:
    task_id: int
    progress: float
    stage: str


@dataclass
class DoneEvent:
    task_id: int
    task_type: TaskType
    result: Any


@dataclass
class ErrorEvent:
    task_id: int
    message: str


class Worker:
    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self._counter = 0
        self._lock = threading.Lock()

    def submit_separation(self, input_path, output_dir,
                          model_name="htdemucs") -> int:
        return self._submit(TaskType.SEPARATE, input_path, output_dir, model_name)

    def submit_separation_bs(self, input_path, output_dir,
                             bs_model="BS-Roformer-SW.ckpt") -> int:
        return self._submit(TaskType.SEPARATE_BS, input_path, output_dir, bs_model)

    def submit_separation_hybrid(self, input_path, output_dir,
                                 demucs_model="htdemucs_ft",
                                 bs_model="BS-Roformer-SW.ckpt") -> int:
        return self._submit(TaskType.SEPARATE_HYBRID, input_path, output_dir,
                            demucs_model, bs_model)

    def submit_transcription(self, input_audio, output_midi,
                             instrument="Фортепиано") -> int:
        return self._submit(TaskType.TRANSCRIBE, input_audio, output_midi, instrument)

    def submit_analysis(self, input_audio, report_dir=None) -> int:
        return self._submit(TaskType.ANALYZE, input_audio, report_dir)

    def submit_pair_analysis(self, vocals_path, instr_path, report_dir=None) -> int:
        return self._submit(TaskType.ANALYZE_PAIR, vocals_path, instr_path, report_dir)

    def submit_piano(self, input_audio, output_midi, model_size="small") -> int:
        return self._submit(TaskType.PIANO, input_audio, output_midi, model_size)

    def submit_midi_analysis(self, midi_path, report_dir=None) -> int:
        return self._submit(TaskType.MIDI_ANALYZE, midi_path, report_dir)

    # ---------- внутреннее ----------
    def _next_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def _submit(self, task_type, *args) -> int:
        task_id = self._next_id()
        t = threading.Thread(
            target=self._run, args=(task_id, task_type, *args), daemon=True
        )
        t.start()
        logger.info("Задача поставлена",
                    extra={"ctx_task_id": task_id, "ctx_type": task_type.name})
        return task_id

    def _run(self, task_id, task_type, *args):
        def progress(p, stage):
            self.events.put(ProgressEvent(task_id, p, stage))

        try:
            if task_type is TaskType.SEPARATE:
                input_path, output_dir, model_name = args
                sep = AudioSeparator(model_name=model_name)
                result = sep.separate(input_path, output_dir, progress_cb=progress)

            elif task_type is TaskType.SEPARATE_BS:
                input_path, output_dir, bs_model = args
                from core.separator_bs import BSAudioSeparator
                sep = BSAudioSeparator(model_name=bs_model)
                result = sep.separate(input_path, output_dir, progress_cb=progress)

            elif task_type is TaskType.SEPARATE_HYBRID:
                input_path, output_dir, demucs_model, bs_model = args
                from core.separator_hybrid import HybridSeparator
                sep = HybridSeparator(demucs_model=demucs_model, bs_model=bs_model)
                result = sep.separate(input_path, output_dir, progress_cb=progress)

            elif task_type is TaskType.TRANSCRIBE:
                input_audio, output_midi, instrument = args
                tr = MidiTranscriber()
                result = tr.transcribe(input_audio, output_midi, instrument,
                                       progress_cb=progress)

            elif task_type is TaskType.ANALYZE:
                input_audio, report_dir = args
                result = analyze_audio(input_audio, report_dir, progress_cb=progress)

            elif task_type is TaskType.ANALYZE_PAIR:
                vocals_path, instr_path, report_dir = args
                from core.super_analyzer import super_analyze_pair
                result = super_analyze_pair(vocals_path, instr_path,
                                            report_dir, progress_cb=progress)

            elif task_type is TaskType.PIANO:
                input_audio, output_midi, model_size = args
                pt = PianoTranscriber(model_size=model_size)
                result = pt.transcribe(input_audio, output_midi, progress_cb=progress)

            elif task_type is TaskType.MIDI_ANALYZE:
                midi_path, report_dir = args
                result = analyze_midi(midi_path, report_dir)

            else:
                raise ValueError(f"Неизвестный тип: {task_type}")

            self.events.put(DoneEvent(task_id, task_type, result))

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Задача упала",
                         extra={"ctx_task_id": task_id, "ctx_tb": tb})
            self.events.put(ErrorEvent(task_id, str(e)))