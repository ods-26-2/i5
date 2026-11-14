import argparse
import json
import logging
import os
import sys
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

try:
    import p2
except ImportError:
    try:
        from src import p2_runtime as p2
    except ImportError:
        import p2_runtime as p2

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]

SAMPLE_RATE: Final = 16000
CHUNK_SECONDS: Final = 30
CHUNK_SAMPLES: Final = SAMPLE_RATE * CHUNK_SECONDS
N_FFT: Final = 400
HOP_LENGTH: Final = 160
N_MELS: Final = 80
PCM16_BYTES: Final = 2
PCM16_SCALE: Final = 32768.0
LOG_FLOOR: Final = 1e-10
LOG_RANGE: Final = 8.0
LOG_SHIFT: Final = 4.0
SLANEY_LINEAR_STEP: Final = 200.0 / 3.0
SLANEY_LOG_START_HZ: Final = 1000.0
SLANEY_LOG_STEP: Final = float(np.log(6.4) / 27.0)
MAX_TOKENS: Final = 448
END_OF_TEXT: Final = "<|endoftext|>"
PROMPT_START: Final = "<|startoftranscript|>"
PROMPT_END: Final = ("<|transcribe|>", "<|notimestamps|>")
DEFAULT_ENCODER: Final = "whisper-small-encoder-1.0.0"
DEFAULT_DECODER: Final = "whisper-small-decoder-1.0.0"
DEFAULT_LANGUAGE: Final = "pt"
DEFAULT_TOKENIZER: Final = Path("models") / "whisper-tokenizer.json"
ENCODER_INPUT: Final = "input_features"
ENCODER_OUTPUT: Final = "last_hidden_state"
DECODER_TOKENS: Final = "input_ids"
DECODER_FEATURES: Final = "encoder_hidden_states"
DECODER_OUTPUT: Final = "logits"
EXIT_OK: Final = 0
EXIT_ERROR: Final = 1


@dataclass(frozen=True, slots=True)
class Tokenizer:
    vocabulary: Mapping[int, str]
    special: Mapping[str, int]

    def prompt(self, language: str) -> list[int]:
        names = (PROMPT_START, f"<|{language}|>", *PROMPT_END)
        return [self.special[name] for name in names]

    @property
    def end_of_text(self) -> int:
        return self.special[END_OF_TEXT]

    def decode(self, tokens: Sequence[int]) -> str:
        decoder = byte_decoder()
        ids = [token for token in tokens if token not in self.special.values()]
        text = "".join(self.vocabulary[token] for token in ids)
        return bytes(decoder[char] for char in text).decode("utf-8", errors="replace").strip()


def byte_decoder() -> dict[str, int]:
    printable = [*range(33, 127), *range(161, 173), *range(174, 256)]
    remaining = [byte for byte in range(256) if byte not in printable]
    mapping = {chr(byte): byte for byte in printable}
    mapping.update({chr(256 + index): byte for index, byte in enumerate(remaining)})
    return mapping


def resolve_tokenizer_path(path: Path) -> Path:
    if path.is_file():
        return path
    cand = Path(__file__).resolve().parent.parent / path
    if cand.is_file():
        return cand
    import glob
    hf = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--openai--whisper-small/snapshots/*/tokenizer.json"))
    if hf and Path(hf[0]).is_file():
        return Path(hf[0])
    return path


def read_tokenizer(path: Path) -> Tokenizer:
    resolved = resolve_tokenizer_path(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    vocabulary = {int(index): token for token, index in data["model"]["vocab"].items()}
    special = {item["content"]: int(item["id"]) for item in data["added_tokens"]}
    vocabulary.update({index: token for token, index in special.items()})
    return Tokenizer(vocabulary=vocabulary, special=special)


def read_wav(path: Path) -> FloatArray:
    with wave.open(str(path), "rb") as audio:
        rate, channels, width = audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
        frames = audio.readframes(audio.getnframes())
    if rate != SAMPLE_RATE or channels != 1 or width != PCM16_BYTES:
        raise ValueError(
            f"áudio deve ser WAV PCM 16 bits, mono, {SAMPLE_RATE} Hz; "
            f"obtido {width * 8} bits, {channels} canais, {rate} Hz"
        )
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return samples / np.float32(PCM16_SCALE)


def _hz_to_mel(hz: FloatArray) -> FloatArray:
    linear = hz / SLANEY_LINEAR_STEP
    start = SLANEY_LOG_START_HZ / SLANEY_LINEAR_STEP
    safe = np.maximum(hz, SLANEY_LOG_START_HZ)
    logarithmic = start + np.log(safe / SLANEY_LOG_START_HZ) / SLANEY_LOG_STEP
    return np.where(hz >= SLANEY_LOG_START_HZ, logarithmic, linear).astype(np.float32)


def _mel_to_hz(mel: FloatArray) -> FloatArray:
    start = SLANEY_LOG_START_HZ / SLANEY_LINEAR_STEP
    linear = mel * SLANEY_LINEAR_STEP
    logarithmic = SLANEY_LOG_START_HZ * np.exp(SLANEY_LOG_STEP * (mel - start))
    return np.where(mel >= start, logarithmic, linear).astype(np.float32)


def mel_filters() -> FloatArray:
    frequencies = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE).astype(np.float32)
    top = _hz_to_mel(np.array([SAMPLE_RATE / 2], dtype=np.float32))[0]
    points = _mel_to_hz(np.linspace(0.0, top, N_MELS + 2, dtype=np.float32))
    lower = (frequencies[None, :] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
    upper = (points[2:, None] - frequencies[None, :]) / (points[2:] - points[1:-1])[:, None]
    triangles = np.maximum(0.0, np.minimum(lower, upper))
    return np.asarray(triangles * (2.0 / (points[2:] - points[:-2]))[:, None], dtype=np.float32)


def log_mel(samples: FloatArray) -> FloatArray:
    audio = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    clipped = samples[:CHUNK_SAMPLES]
    audio[: clipped.size] = clipped
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    power = np.abs(np.fft.rfft(frames * window, axis=1)[:-1]) ** 2
    mel = mel_filters() @ power.T
    logs = np.log10(np.maximum(mel, LOG_FLOOR))
    logs = np.maximum(logs, logs.max() - LOG_RANGE)
    return np.asarray((logs + LOG_SHIFT) / LOG_SHIFT, dtype=np.float32)[np.newaxis]


def generate(
    encoder: p2.Handle,
    decoder: p2.Handle,
    samples: FloatArray,
    *,
    tokenizer: Tokenizer,
    language: str,
) -> tuple[list[int], list[float]]:
    outputs, _ = p2.infer(encoder, {ENCODER_INPUT: log_mel(samples)})
    features = outputs[ENCODER_OUTPUT]
    prompt = tokenizer.prompt(language)
    tokens = list(prompt)
    probabilities: list[float] = []
    while len(tokens) < MAX_TOKENS:
        ids = np.array([tokens], dtype=np.int64)
        step, _ = p2.infer(decoder, {DECODER_TOKENS: ids, DECODER_FEATURES: features})
        logits = np.asarray(step[DECODER_OUTPUT], dtype=np.float32)[0, -1]
        token = int(logits.argmax())
        if token == tokenizer.end_of_text:
            break
        tokens.append(token)
        probabilities.append(_probability(logits, token))
    return tokens[len(prompt) :], probabilities


def transcribe(
    encoder: p2.Handle,
    decoder: p2.Handle,
    samples: FloatArray,
    *,
    tokenizer: Tokenizer,
    language: str,
) -> str:
    tokens, _ = generate(encoder, decoder, samples, tokenizer=tokenizer, language=language)
    return tokenizer.decode(tokens)


def _probability(logits: FloatArray, token: int) -> float:
    weights = np.exp(logits - logits.max())
    return float(weights[token] / weights.sum())


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Transcrição com Whisper em duas etapas no p2.")
    parser.add_argument("audio", type=Path, help="WAV PCM 16 bits, mono, 16 kHz")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--decoder", default=DEFAULT_DECODER)
    args = parser.parse_args(argv)
    audio: object = args.audio
    tokenizer_path: object = args.tokenizer
    language: object = args.language
    encoder_ref: object = args.encoder
    decoder_ref: object = args.decoder
    if not isinstance(audio, Path) or not isinstance(tokenizer_path, Path):
        parser.error("caminhos inválidos")
    if not (
        isinstance(language, str) and isinstance(encoder_ref, str) and isinstance(decoder_ref, str)
    ):
        parser.error("argumentos inválidos")
    try:
        samples = read_wav(audio)
        tokenizer = read_tokenizer(tokenizer_path)
        with p2.load(encoder_ref) as encoder, p2.load(decoder_ref) as decoder:
            logger.info(
                "%s", transcribe(encoder, decoder, samples, tokenizer=tokenizer, language=language)
            )
    except (p2.P2Error, ValueError, KeyError, OSError) as err:
        logger.error("%s", err)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
