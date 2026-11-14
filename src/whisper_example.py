import argparse
import json
import logging
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

import numpy as np

try:
    import p2
except ImportError:
    try:
        from src import p2_runtime as p2
    except ImportError:
        import p2_runtime as p2
try:
    from whisper_demo import (
        CHUNK_SAMPLES,
        DEFAULT_DECODER,
        DEFAULT_ENCODER,
        DEFAULT_LANGUAGE,
        DEFAULT_TOKENIZER,
        FloatArray,
        Tokenizer,
        generate,
        read_tokenizer,
        read_wav,
    )
except ImportError:
    from src.whisper_demo import (
        CHUNK_SAMPLES,
        DEFAULT_DECODER,
        DEFAULT_ENCODER,
        DEFAULT_LANGUAGE,
        DEFAULT_TOKENIZER,
        FloatArray,
        Tokenizer,
        generate,
        read_tokenizer,
        read_wav,
    )

logger = logging.getLogger(__name__)

PRECISION_OPTION: Final = "precision"
MS_PER_SECOND: Final = 1000.0
EXIT_OK: Final = 0
EXIT_ERROR: Final = 1


class Transcript(TypedDict):
    text: str
    confidence: float
    token_probs: list[float]


class Metrics(TypedDict):
    latency_ms: float
    gpu_memory_mb: float


@dataclass(frozen=True, slots=True)
class BridgeHandle:
    encoder: p2.Handle
    decoder: p2.Handle
    memory_mb: float


class WhisperBridge:
    def __init__(
        self,
        *,
        encoder_ref: str = DEFAULT_ENCODER,
        decoder_ref: str = DEFAULT_DECODER,
        tokenizer_path: Path = DEFAULT_TOKENIZER,
        language: str = DEFAULT_LANGUAGE,
    ) -> None:
        self._encoder_ref = encoder_ref
        self._decoder_ref = decoder_ref
        self._tokenizer: Tokenizer = read_tokenizer(tokenizer_path)
        self._language = language

    def load(
        self, engine_ref: str = "", options: Mapping[str, object] | None = None
    ) -> BridgeHandle:
        engine_options = _engine_options(options)
        encoder = p2.load(self._encoder_ref, engine_options)
        try:
            decoder = p2.load(self._decoder_ref, engine_options)
        except p2.P2Error:
            p2.release(encoder)
            raise
        memory_mb = encoder.memory_mb + decoder.memory_mb
        return BridgeHandle(encoder=encoder, decoder=decoder, memory_mb=memory_mb)

    def info(self, handle: BridgeHandle) -> p2.EngineInfo:
        return p2.info(handle.decoder)

    def infer(self, handle: BridgeHandle, audio: object) -> tuple[Transcript, Metrics]:
        started = time.perf_counter()
        samples = _as_samples(audio)
        texts: list[str] = []
        probabilities: list[float] = []
        for start in range(0, samples.size, CHUNK_SAMPLES):
            chunk = samples[start : start + CHUNK_SAMPLES]
            tokens, chunk_probabilities = generate(
                handle.encoder,
                handle.decoder,
                chunk,
                tokenizer=self._tokenizer,
                language=self._language,
            )
            texts.append(self._tokenizer.decode(tokens))
            probabilities.extend(chunk_probabilities)
        transcript: Transcript = {
            "text": " ".join(text for text in texts if text),
            "confidence": round(float(np.mean(probabilities)), 4) if probabilities else 0.0,
            "token_probs": probabilities,
        }
        latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
        return transcript, {"latency_ms": latency_ms, "gpu_memory_mb": handle.memory_mb}

    def release(self, handle: BridgeHandle) -> None:
        p2.release(handle.decoder)
        p2.release(handle.encoder)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Áudio entra, texto sai, sobre a API do p2.")
    parser.add_argument("audio", type=Path, help="WAV PCM 16 bits, mono, 16 kHz")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    args = parser.parse_args(argv)
    audio: object = args.audio
    tokenizer: object = args.tokenizer
    language: object = args.language
    if not isinstance(audio, Path) or not isinstance(tokenizer, Path):
        parser.error("caminhos inválidos")
    if not isinstance(language, str):
        parser.error("--language inválido")
    try:
        bridge = WhisperBridge(tokenizer_path=tokenizer, language=language)
        handle = bridge.load()
        try:
            transcript, metrics = bridge.infer(handle, read_wav(audio))
        finally:
            bridge.release(handle)
    except (p2.P2Error, ValueError, KeyError, OSError) as err:
        logger.error("%s", err)
        return EXIT_ERROR
    summary = {"text": transcript["text"], "confidence": transcript["confidence"], **metrics}
    logger.info("%s", json.dumps(summary, ensure_ascii=False))
    return EXIT_OK


def _engine_options(options: Mapping[str, object] | None) -> dict[str, object] | None:
    if options is None or PRECISION_OPTION not in options:
        return None
    return {PRECISION_OPTION: options[PRECISION_OPTION]}


def _as_samples(audio: object) -> FloatArray:
    if isinstance(audio, Mapping):
        audio = audio.get("audio")
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise p2.P2Error(p2.ErrorCode.INVALID_INPUT_SHAPE, "áudio vazio")
    return samples


if __name__ == "__main__":
    raise SystemExit(main())
