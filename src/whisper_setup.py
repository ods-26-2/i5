import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import onnx

from p2.manifest import resolve_engine_root

logger = logging.getLogger(__name__)

HF_BASE: Final = "https://huggingface.co/onnx-community/whisper-small/resolve/main"
SAMPLE_URL: Final = (
    "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav"
)
HASH_CHUNK_BYTES: Final = 1 << 20
ARTIFACT_VERSION: Final = "1.0.0"
DEFAULT_MODELS_DIR: Final = Path("models")
DEFAULT_PRECISION: Final = "FP16"
EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
MEL_SHAPE: Final = [1, 80, 3000]
FEATURES_SHAPE: Final = [1, 1500, 768]
MAX_TOKENS: Final = 448
OPT_TOKENS: Final = 64
ENCODER_ONNX: Final = "whisper-small-encoder.onnx"
DECODER_ONNX: Final = "whisper-small-decoder.onnx"
DECODER_LOGITS_ONNX: Final = "whisper-small-decoder-logits.onnx"
DECODER_OUTPUTS: Final = frozenset({"logits"})
TOKENIZER_FILE: Final = "whisper-tokenizer.json"
SAMPLE_FILE: Final = "whisper-sample.wav"
RAW_SAMPLE_FILE: Final = "whisper-sample-44k.wav"


@dataclass(frozen=True, slots=True)
class RemoteFile:
    url: str
    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Stage:
    artifact_id: str
    task: str
    onnx_name: str
    input_shape: Mapping[str, list[int]]
    dynamic_shapes: Mapping[str, Mapping[str, list[int]]]


def fixed_profile(shape: list[int]) -> dict[str, list[int]]:
    return {"min": shape, "opt": shape, "max": shape}


ENCODER: Final = Stage(
    artifact_id="whisper-small-encoder",
    task="speech-encoder",
    onnx_name=ENCODER_ONNX,
    input_shape={"input_features": [-1, -1, -1]},
    dynamic_shapes={"input_features": fixed_profile(MEL_SHAPE)},
)
DECODER: Final = Stage(
    artifact_id="whisper-small-decoder",
    task="speech-decoder",
    onnx_name=DECODER_LOGITS_ONNX,
    input_shape={"input_ids": [-1, -1], "encoder_hidden_states": [-1, -1, 768]},
    dynamic_shapes={
        "input_ids": {"min": [1, 1], "opt": [1, OPT_TOKENS], "max": [1, MAX_TOKENS]},
        "encoder_hidden_states": fixed_profile(FEATURES_SHAPE),
    },
)
FILES: Final = (
    RemoteFile(
        f"{HF_BASE}/onnx/encoder_model.onnx",
        ENCODER_ONNX,
        "b37cd6625dc36f9178ec7539a1876b9680ea26a910097e092be39dc766320c7b",
    ),
    RemoteFile(
        f"{HF_BASE}/onnx/decoder_model.onnx",
        DECODER_ONNX,
        "12130ce1e82372a8e54e753d3fb4339f289470cbbe2eceb9c8bf89cb2cc6fe63",
    ),
    RemoteFile(
        f"{HF_BASE}/tokenizer.json",
        TOKENIZER_FILE,
        "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566",
    ),
    RemoteFile(
        SAMPLE_URL,
        RAW_SAMPLE_FILE,
        "aa81c2552465568567e670f3823117e633900d16bd6202346a72f3c8464c74c8",
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(remote: RemoteFile, directory: Path) -> Path:
    target = directory / remote.name
    if target.is_file() and sha256_of(target) == remote.sha256:
        logger.info("já baixado: %s", target)
        return target
    logger.info("baixando %s", remote.url)
    command = ("curl", "-fL", "--http1.1", "--retry", "5", "-C", "-", "-o", str(target), remote.url)
    subprocess.run(command, check=True)
    actual = sha256_of(target)
    if actual != remote.sha256:
        raise ValueError(f"hash de {target}: esperado {remote.sha256}, obtido {actual}")
    return target


def convert_sample(directory: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg ausente: áudio de teste não convertido para 16 kHz mono")
        return
    source, target = directory / RAW_SAMPLE_FILE, directory / SAMPLE_FILE
    command = (ffmpeg, "-loglevel", "error", "-y", "-i", str(source), "-ar", "16000", "-ac", "1")
    subprocess.run((*command, "-c:a", "pcm_s16le", str(target)), check=True)


def keep_outputs(source: Path, target: Path, names: frozenset[str]) -> Path:
    if target.is_file():
        return target
    logger.info("removendo as saídas sem uso de %s", source.name)
    model = onnx.load(str(source))
    kept = [output for output in model.graph.output if output.name in names]
    del model.graph.output[:]
    model.graph.output.extend(kept)
    onnx.save(model, str(target))
    return target


def write_package(directory: Path, stage: Stage, onnx_path: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    link = directory / onnx_path.name
    link.unlink(missing_ok=True)
    link.symlink_to(onnx_path.resolve())
    metadata = {
        "artifact_id": stage.artifact_id,
        "artifact_version": ARTIFACT_VERSION,
        "onnx_file": onnx_path.name,
        "hash_sha256": sha256_of(onnx_path),
        "precision": "FP32",
        "task": stage.task,
        "input_shape": dict(stage.input_shape),
        "dynamic_shapes": dict(stage.dynamic_shapes),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def build_arguments(package: Path, args: argparse.Namespace) -> list[str]:
    arguments = [str(package)]
    arguments += ["--cpu-only"] if args.cpu_only else ["--precision", *args.precision]
    if args.force:
        arguments.append("--force")
    if args.engine_dir is not None:
        arguments += ["--engine-dir", str(args.engine_dir)]
    return arguments


def compile_stage(package: Path, stage: Stage, args: argparse.Namespace) -> int:
    engine_ref = f"{stage.artifact_id}-{ARTIFACT_VERSION}"
    target = resolve_engine_root(args.engine_dir, os.environ) / engine_ref
    if target.exists() and not args.force:
        logger.info("já compilado: %s (use --force para refazer)", target)
        return EXIT_OK
    command = (sys.executable, "-m", "p2.cli.build", *build_arguments(package, args))
    return subprocess.run(command, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Baixa, empacota e compila o Whisper-small.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--engine-dir", type=Path, help="raiz dos engines")
    parser.add_argument("--precision", nargs="+", default=[DEFAULT_PRECISION])
    parser.add_argument("--cpu-only", action="store_true", help="prepara para rodar na CPU")
    parser.add_argument("--force", action="store_true", help="substitui engines existentes")
    args = parser.parse_args(argv)
    models_dir: object = args.models_dir
    if not isinstance(models_dir, Path):
        parser.error("--models-dir inválido")
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        for remote in FILES:
            download(remote, models_dir)
        convert_sample(models_dir)
        keep_outputs(models_dir / DECODER_ONNX, models_dir / DECODER_LOGITS_ONNX, DECODER_OUTPUTS)
    except (OSError, ValueError, subprocess.CalledProcessError) as err:
        logger.error("%s", err)
        return EXIT_ERROR
    for name, stage in (("encoder", ENCODER), ("decoder", DECODER)):
        package = write_package(
            models_dir / "whisper-packages" / name, stage, models_dir / stage.onnx_name
        )
        if compile_stage(package, stage, args) != EXIT_OK:
            return EXIT_ERROR
    logger.info("pronto: python3 examples/whisper_demo.py <audio.wav> --language pt")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
