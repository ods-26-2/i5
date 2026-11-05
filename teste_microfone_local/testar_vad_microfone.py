#!/usr/bin/env python3
"""
=============================================================================
[ENTRADA DE MICROFONE LOCAL SUBSTITUINDO APENAS O P5 (ods.p5.audio_chunk)]
=============================================================================
Este script substitui SOMENTE a origem de áudio (P5) pela captura contínua
do seu microfone local (16.000 Hz mono) e entrega o stream diretamente para o
`RealtimeSpeechPipeline` oficial do I5 (`src/pipeline.py`).

Tudo o mais roda 100% com a implementação oficial do repositório I5:
  1. `AudioPreprocessor` (src/audio_preprocessor.py)
  2. `VADSegmenter` - Silero VAD (src/vad_segmenter.py)
  3. `KeywordSpotter` - Wake Word da Ester (src/keyword_spotter.py)
  4. `TranscritorWhisper` - Whisper ASR (src/TranscriptionWhisper.py)
  5. `TranscriptKafkaPublisher` - Publicação no Kafka `ods.i5.transcript`
     (src/kafka_publisher.py) e salvamento em `jsons_gerados/`.
=============================================================================
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator

# Adiciona a raiz do projeto I5 ao sys.path para importar os módulos oficiais de `src/`
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config_loader import load_pipeline_config
from src.kafka_publisher import TranscriptKafkaPublisher
from src.pipeline import RealtimeSpeechPipeline


def mic_audio_stream(
    target_sample_rate: int = 16000,
    chunk_size_samples: int = 1600,  # 100ms a 16kHz (mesmo chunk_size padrão do pipeline.py)
) -> Generator[np.ndarray, None, None]:
    """
    Substitui o `simulate_audio_stream` / `KafkaAudioConsumer` (P5) lendo
    continuamente do microfone local em 16.000 Hz mono float32.
    """
    bytes_per_sample = 2  # PCM 16-bit signed (s16le)
    chunk_bytes = chunk_size_samples * bytes_per_sample

    if shutil.which("parecord"):
        cmd = [
            "parecord",
            "--raw",
            f"--rate={target_sample_rate}",
            "--format=s16le",
            "--channels=1",
            "--latency-msec=50",
        ]
        fonte = "parecord (PipeWire/PulseAudio)"
    elif shutil.which("arecord"):
        cmd = [
            "arecord",
            "-q",
            "-t", "raw",
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(target_sample_rate),
        ]
        fonte = "arecord (ALSA)"
    else:
        raise RuntimeError("Nenhum utilitário de captura ('parecord' ou 'arecord') encontrado.")

    print(f"[Microfone -> I5] Capturando áudio contínuo via {fonte} a {target_sample_rate} Hz...")
    print("[Microfone -> I5] Fale no microfone (Pressione Ctrl+C para encerrar).\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=chunk_bytes * 2,
    )

    try:
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw or len(raw) < chunk_bytes:
                break
            pcm_int16 = np.frombuffer(raw, dtype=np.int16)
            chunk_float32 = pcm_int16.astype(np.float32) / 32768.0
            yield chunk_float32
    except KeyboardInterrupt:
        print("\n[Microfone -> I5] Captura encerrada pelo usuário (Ctrl+C). Finalizando pipeline...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    parser = argparse.ArgumentParser(
        description="Entrada contínua de Microfone para o Pipeline Oficial I5 (substituindo apenas o P5)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "pipeline_config.yaml"),
        help="Caminho para o arquivo YAML de configuração (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--origem",
        type=str,
        default="mic_balcao_1",
        help="Identificador da origem/microfone no payload (default: mic_balcao_1)",
    )
    parser.add_argument(
        "--publish-kafka",
        action="store_true",
        default=True,
        help="Publica os eventos transcritos no tópico Kafka ods.i5.transcript (ativo por padrão)",
    )
    parser.add_argument(
        "--no-publish-kafka",
        action="store_false",
        dest="publish_kafka",
        help="Desativa publicação no Kafka ods.i5.transcript caso queira testar offline",
    )
    parser.add_argument(
        "--kafka-broker",
        type=str,
        default=None,
        help="Endereço do broker Kafka (default: 34.66.235.91:9093 ou KAFKA_BOOTSTRAP_SERVERS)",
    )
    parser.add_argument(
        "--kafka-topic",
        type=str,
        default=None,
        help="Tópico Kafka de saída (default: ods.i5.transcript)",
    )
    parser.add_argument(
        "--wake-word",
        action="store_true",
        default=True,
        help="Ativa o KeywordSpotter oficial da Ester (ativo por padrão)",
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_false",
        dest="wake_word",
        help="Desativa o filtro de Wake Word (Modo Contínuo)",
    )
    parser.add_argument(
        "--wake-words",
        type=str,
        default="lexie",
        help="Termo de ativação configurado no KeywordSpotter (default: lexie)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Limiar do VAD para ativação de fala (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--silence-threshold-ms",
        type=int,
        default=None,
        help="Limiar de silêncio para encerramento de batch em ms (sobrescreve o YAML)",
    )
    args = parser.parse_args()

    # 1. Carrega exatamente a mesma configuração YAML oficial do pipeline.py
    config = load_pipeline_config(args.config)
    if args.threshold is not None:
        config["vad"]["speech_threshold"] = args.threshold
    if args.silence_threshold_ms is not None:
        config["vad"]["silence_threshold_ms"] = args.silence_threshold_ms

    sample_rate = config["audio"].get("sample_rate", 16000)
    chunk_ms = config["audio"].get("chunk_duration_ms", 100)

    # 2. Instancia o publicador Kafka oficial do I5 (ods.i5.transcript)
    kafka_pub = None
    if args.publish_kafka:
        kafka_pub = TranscriptKafkaPublisher(
            bootstrap_servers=args.kafka_broker,
            topic=args.kafka_topic,
        )
        print(f"[I5 - Kafka] Publicando em {kafka_pub.bootstrap_servers} -> Tópico: {kafka_pub.topic}")

    wake_words_list = [w.strip() for w in args.wake_words.split(",") if w.strip()]

    print(f"\n=======================================================")
    print(f"   Iniciando Pipeline Oficial I5: VAD + KWS + Whisper")
    print(f"   Fonte de Áudio (substituindo P5): Microfone Local")
    print(f"   Filtro Wake Word (Ester - CPU): {'ATIVO (' + ', '.join(wake_words_list) + ')' if args.wake_word else 'DESATIVADO (Modo Contínuo)'}")
    print(f"   VAD Speech Threshold: {config['vad'].get('speech_threshold')}")
    print(f"   VAD Silence Threshold: {config['vad'].get('silence_threshold_ms')}ms")
    print(f"=======================================================\n")

    # 3. Instancia o RealtimeSpeechPipeline oficial sem nenhuma alteração
    pipeline = RealtimeSpeechPipeline(
        config=config,
        origem=args.origem,
        kafka_publisher=kafka_pub,
        enable_wake_word=args.wake_word,
        wake_words=wake_words_list,
    )

    # 4. Alimenta o process_stream oficial com os chunks vindos do microfone
    chunk_size = int(sample_rate * (chunk_ms / 1000.0))
    stream = mic_audio_stream(
        target_sample_rate=sample_rate,
        chunk_size_samples=chunk_size,
    )

    start_total = time.perf_counter()
    resultados = pipeline.process_stream(stream)
    elapsed = time.perf_counter() - start_total

    total_vad = pipeline.batches_dispatched_to_gpu + pipeline.batches_filtered_by_kws
    print(f"\n=======================================================")
    print(f"   Processamento Concluído!")
    print(f"   Tempo total: {elapsed:.2f}s")
    print(f"   Lotes detectados pelo VAD (CPU): {total_vad}")
    if args.wake_word:
        print(f"   Lotes bloqueados pelo KWS/Wake Word (CPU): {pipeline.batches_filtered_by_kws}")
    print(f"   Total de frases/batches transcritos na GPU: {len(resultados)}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
