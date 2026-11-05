"""
VAD fallback baseado em limiar de energia RMS e silêncio.
Opera com precisão quando dependências C (PyTorch/ONNX) estiverem indisponíveis no ambiente local.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, List, Optional
import numpy as np


class VADState(Enum):
    SILENCE = "SILENCE"
    SPEAKING = "SPEAKING"


@dataclass
class AudioBatch:
    batch_index: int
    audio: np.ndarray
    start_time: float
    end_time: float
    duration: float
    sample_rate: int = 16000
    metadata: dict = field(default_factory=dict)


class EnergyVADSegmenter:
    CHUNK_SIZE = 512

    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        silence_threshold_ms: int = 700,
        pre_speech_pad_ms: int = 250,
        post_speech_pad_ms: int = 200,
        min_speech_duration_ms: int = 250,
        use_onnx: bool = True,
    ):
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.silence_threshold_samples = int(sample_rate * silence_threshold_ms / 1000)
        self.pre_pad_samples = int(sample_rate * pre_speech_pad_ms / 1000)
        self.post_pad_samples = int(sample_rate * post_speech_pad_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)

        self.max_pre_roll_chunks = max(1, int(np.ceil(self.pre_pad_samples / self.CHUNK_SIZE)))
        self.post_pad_chunks = int(np.ceil(self.post_pad_samples / self.CHUNK_SIZE))

        self.state = VADState.SILENCE
        self.pre_roll_buffer: Deque[np.ndarray] = deque(maxlen=self.max_pre_roll_chunks)
        self.current_speech_chunks: List[np.ndarray] = []
        self.last_speech_chunk_idx: int = 0
        self.consecutive_silence_samples: int = 0
        self.total_processed_samples: int = 0
        self.batch_start_sample: int = 0
        self.batch_counter: int = 0
        self.first_captured_at: Optional[datetime] = None
        self.leftover_samples: np.ndarray = np.empty(0, dtype=np.float32)

    def push_audio(self, audio_chunk: np.ndarray, captured_at: Optional[Any] = None) -> List[AudioBatch]:
        batches: List[AudioBatch] = []
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()

        if self.leftover_samples.size > 0:
            audio_chunk = np.concatenate([self.leftover_samples, audio_chunk])
            self.leftover_samples = np.empty(0, dtype=np.float32)

        total_samples = len(audio_chunk)
        num_full_chunks = total_samples // self.CHUNK_SIZE
        if num_full_chunks == 0:
            self.leftover_samples = audio_chunk.copy()
            return batches

        for i in range(num_full_chunks):
            start = i * self.CHUNK_SIZE
            end = start + self.CHUNK_SIZE
            chunk = audio_chunk[start:end]

            rms = float(np.sqrt(np.mean(np.square(chunk))))
            is_speech = rms >= 0.012

            if self.state == VADState.SILENCE:
                if is_speech:
                    self.state = VADState.SPEAKING
                    self.batch_start_sample = max(
                        0,
                        self.total_processed_samples - len(self.pre_roll_buffer) * self.CHUNK_SIZE
                    )
                    self.current_speech_chunks = list(self.pre_roll_buffer)
                    self.current_speech_chunks.append(chunk)
                    self.last_speech_chunk_idx = len(self.current_speech_chunks)
                    self.consecutive_silence_samples = 0
                else:
                    self.pre_roll_buffer.append(chunk)
            elif self.state == VADState.SPEAKING:
                self.current_speech_chunks.append(chunk)
                if is_speech:
                    self.consecutive_silence_samples = 0
                    self.last_speech_chunk_idx = len(self.current_speech_chunks)
                else:
                    self.consecutive_silence_samples += self.CHUNK_SIZE
                    if self.consecutive_silence_samples >= self.silence_threshold_samples:
                        batch = self._create_batch()
                        if batch:
                            batches.append(batch)
                        self.state = VADState.SILENCE
                        self.current_speech_chunks.clear()
                        self.pre_roll_buffer.clear()
                        self.consecutive_silence_samples = 0

            self.total_processed_samples += self.CHUNK_SIZE

        rem = total_samples % self.CHUNK_SIZE
        if rem > 0:
            self.leftover_samples = audio_chunk[-rem:].copy()

        return batches

    def _create_batch(self) -> Optional[AudioBatch]:
        if not self.current_speech_chunks:
            return None
        keep = min(len(self.current_speech_chunks), self.last_speech_chunk_idx + self.post_pad_chunks)
        kept = self.current_speech_chunks[:keep]
        if not kept:
            return None
        batch_audio = np.concatenate(kept)
        if len(batch_audio) < self.min_speech_samples:
            return None
        start_time = self.batch_start_sample / self.sample_rate
        duration = len(batch_audio) / self.sample_rate
        end_time = start_time + duration
        self.batch_counter += 1
        return AudioBatch(
            batch_index=self.batch_counter,
            audio=batch_audio,
            start_time=round(start_time, 3),
            end_time=round(end_time, 3),
            duration=round(duration, 3),
            sample_rate=self.sample_rate,
        )

    def flush(self) -> Optional[AudioBatch]:
        if self.state == VADState.SPEAKING and self.current_speech_chunks:
            batch = self._create_batch()
            self.state = VADState.SILENCE
            self.current_speech_chunks.clear()
            self.pre_roll_buffer.clear()
            return batch
        return None
