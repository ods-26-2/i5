#!/usr/bin/env python3
"""
Pipeline de Streaming e Transcrição em Tempo Real (Componente I5).

Orquestra o fluxo contínuo de áudio:
1. Ingestão de áudio em streaming (P5 Mock / Microfone) a 16.000 Hz.
2. Pré-processamento e filtragem passa-banda (AudioPreprocessor na CPU).
3. Detecção de atividade de voz e segmentação contínua (VADSegmenter na CPU).
4. Fila desacoplada em memória RAM (Queue Produtor-Consumidor).
5. Transcrição forçada em Português com timestamps preservados (Whisper na GPU).
6. Emissão de payloads estruturados para A6/A7 (via callback ou stdout).
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Callable, Generator, List, Optional

# Suprime avisos de depreciação do Transformers/PyTorch
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError:
    librosa = None

try:
    import numpy as np
except ImportError:
    np = None

# Adiciona o diretório raiz ao sys.path para imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.audio_preprocessor import AudioPreprocessor
except ImportError:
    AudioPreprocessor = None

from src.kafka_publisher import (
    DEFAULT_TOPIC,
    TranscriptKafkaPublisher,
    construir_envelope_transcricao,
)
from src.keyword_spotter import DEFAULT_WAKE_WORDS, KeywordSpotter, WakeWordResult

try:
    from src.vad_segmenter import AudioBatch, VADSegmenter
except ImportError:
    from dataclasses import dataclass, field

    VADSegmenter = None

    @dataclass
    class AudioBatch:
        batch_index: int
        audio: object
        start_time: float
        end_time: float
        duration: float
        sample_rate: int = 16000
        metadata: dict = field(default_factory=dict)

try:
    from src.TranscriptionWhisper import TranscritorWhisper
except ImportError:
    from TranscriptionWhisper import TranscritorWhisper


class RealtimeSpeechPipeline:
    """
    Orquestrador do Componente I5 para processamento contínuo de áudio e transcrição.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        silence_threshold_ms: int = 700,
        apply_noise_filter: bool = True,
        on_transcription: Optional[Callable[[dict], None]] = None,
        origem: str = "mic_balcao_1",
        kafka_publisher: Optional[TranscriptKafkaPublisher] = None,
        enable_wake_word: bool = False,
        continuous_mode: Optional[bool] = None,
        wake_words: Optional[List[str]] = None,
        keyword_spotter: Optional[KeywordSpotter] = None,
        preprocessor: Optional[object] = None,
        segmenter: Optional[object] = None,
        transcritor: Optional[object] = None,
    ):
        """
        Inicializa o pipeline com os componentes de VAD, Keyword Spotting (CPU),
        Transcrição (GPU/P2) e Publicação Kafka (B2).
        """
        self.sample_rate = sample_rate
        self.apply_noise_filter = apply_noise_filter
        self.on_transcription = on_transcription
        self.origem = origem
        self.kafka_publisher = kafka_publisher

        # Se `continuous_mode` for informado explicitamente:
        # - continuous_mode=True  -> desativa filtro de wake word (despacha todos os lotes do VAD)
        # - continuous_mode=False -> exige detecção do termo de ativação antes de enviar à GPU
        if continuous_mode is not None:
            self.enable_wake_word = not bool(continuous_mode)
            self.continuous_mode = bool(continuous_mode)
        elif keyword_spotter is not None:
            self.enable_wake_word = bool(keyword_spotter.enabled)
            self.continuous_mode = not self.enable_wake_word
        else:
            self.enable_wake_word = bool(enable_wake_word)
            self.continuous_mode = not self.enable_wake_word

        self.device = device or "cuda"

        print(f"[I5 - Pipeline] Inicializando componentes...")
        print(f"[I5 - Pipeline] Runtime de inferência P2 (device alvo: {self.device})")

        # Front-End de Áudio (CPU)
        # Desativa subtração espectral em streaming curto para evitar canibalização da fala
        if preprocessor is not None:
            self.preprocessor = preprocessor
        elif AudioPreprocessor is not None:
            self.preprocessor = AudioPreprocessor(
                target_sample_rate=self.sample_rate,
                enable_spectral_subtraction=False,
            )
        else:
            self.preprocessor = None

        if segmenter is not None:
            self.segmenter = segmenter
        elif VADSegmenter is not None:
            self.segmenter = VADSegmenter(
                sample_rate=self.sample_rate,
                speech_threshold=speech_threshold,
                silence_threshold_ms=silence_threshold_ms,
                use_onnx=True,
            )
        else:
            self.segmenter = None

        # Filtro leve de Keyword Spotting (Wake Word) na CPU antes do ASR Whisper
        if keyword_spotter is not None:
            self.keyword_spotter = keyword_spotter
            self.keyword_spotter.enabled = self.enable_wake_word
        else:
            self.keyword_spotter = KeywordSpotter(
                keywords=wake_words or list(DEFAULT_WAKE_WORDS),
                sample_rate=self.sample_rate,
                enabled=self.enable_wake_word,
            )

        # Transcritor ASR (GPU via P2)
        if transcritor is not None:
            self.transcritor = transcritor
        else:
            self.transcritor = TranscritorWhisper(
                model_name=model_name,
                device=self.device,
            )

        # Fila desacoplada em RAM entre VAD+KWS (CPU) e Whisper (GPU)
        self.batch_queue: queue.Queue = queue.Queue()
        self.results: List[dict] = []
        self.rejected_batches: List[dict] = []
        self.batches_dispatched_to_gpu: int = 0
        self.batches_filtered_by_kws: int = 0
        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None

    def _transcription_worker(self) -> None:
        """
        Consumidor executado em thread dedicada para inferência no Whisper.
        Lê batches de fala da fila e processa diretamente da memória RAM.
        """
        while self._running or not self.batch_queue.empty():
            try:
                item = self.batch_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                # Sinalizador de término (poison pill)
                self.batch_queue.task_done()
                break

            batch: AudioBatch = item
            inference_start = time.perf_counter()

            # Transcreve o array em memória (sem I/O de disco)
            resultado = self.transcritor.transcrever_audio(
                batch.audio,
                sampling_rate=batch.sample_rate,
                origem=self.origem,
            )
            inference_time_ms = (time.perf_counter() - inference_start) * 1000

            if isinstance(resultado, str):
                envelope = json.loads(resultado)
                pl = envelope.get("payload", {})
                texto_extraido = pl.get("texto_transcrito", "")
                confianca_extraida = pl.get("confianca_geral", pl.get("confidence", 0.0))
                uncertain_flag = pl.get("uncertain", bool(confianca_extraida < 0.5))
            else:
                texto_extraido = resultado.get("texto", "")
                confianca_extraida = resultado.get("confianca", 0.0)
                envelope = construir_envelope_transcricao(
                    texto_transcrito=texto_extraido,
                    confianca=confianca_extraida,
                    origem=self.origem,
                    schema=self.kafka_publisher.topic if self.kafka_publisher else DEFAULT_TOPIC,
                )
                uncertain_flag = envelope["payload"]["uncertain"]

            # Publica no barramento Kafka (tópico ods.i5.transcript) se o produtor estiver ativo
            if self.kafka_publisher is not None:
                self.kafka_publisher.publish_envelope(envelope, flush=False)

            # Monta o contrato de saída alinhado ao CONTEXT.md + Envelope B2
            payload = {
                "batch_index": batch.batch_index,
                "start_time": round(batch.start_time, 3),
                "end_time": round(batch.end_time, 3),
                "duration": round(batch.duration, 3),
                "texto": texto_extraido,
                "confianca": confianca_extraida,
                "uncertain": uncertain_flag,
                "envelope": envelope,
                "inference_time_ms": round(inference_time_ms, 2),
                "timestamp_criacao": time.time(),
                "metadata": batch.metadata,
            }

            self.results.append(payload)

            if self.on_transcription:
                self.on_transcription(payload)
            else:
                print(
                    f"[{payload['start_time']:6.2f}s -> {payload['end_time']:6.2f}s] "
                    f"(conf: {payload['confianca']:.2f} | uncertain: {payload['uncertain']} | {payload['inference_time_ms']:.0f}ms) "
                    f"=> \"{payload['texto']}\""
                )

            self.batch_queue.task_done()

    def start(self) -> None:
        """
        Inicia a thread consumidora da GPU.
        """
        self._running = True
        self.results.clear()
        self._consumer_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="WhisperWorker",
        )
        self._consumer_thread.start()

    def stop(self) -> None:
        """
        Encerra a thread consumidora, aguarda a finalização dos batches pendentes,
        libera o engine na GPU via P2 (`release`) e faz flush do produtor Kafka (B2).
        """
        self._running = False
        self.batch_queue.put(None)  # Envia sinalizador de término
        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join()
        if hasattr(self.transcritor, "release"):
            self.transcritor.release()
        if self.kafka_publisher is not None:
            self.kafka_publisher.flush()

    def dispatch_batch_if_activated(self, batch: AudioBatch) -> bool:
        """
        Avalia na CPU se o lote detectado pelo VAD inicia com o termo de ativação
        (ou se o modo contínuo está ativo) antes de despachá-lo para a fila da GPU.

        Returns:
            True se o lote foi enfileirado para inferência no Whisper (GPU);
            False se foi bloqueado pelo filtro de Keyword Spotting na CPU.
        """
        self.keyword_spotter.enabled = self.enable_wake_word
        kws_result: WakeWordResult = self.keyword_spotter.check_batch(batch)

        if not hasattr(batch, "metadata") or batch.metadata is None:
            batch.metadata = {}

        batch.metadata["kws_checked"] = True
        batch.metadata["kws_detected"] = kws_result.detected
        batch.metadata["kws_keyword"] = kws_result.matched_keyword
        batch.metadata["kws_confidence"] = kws_result.confidence
        batch.metadata["kws_reason"] = kws_result.reason

        if kws_result.detected:
            self.batches_dispatched_to_gpu += 1
            self.batch_queue.put(batch)
            return True

        self.batches_filtered_by_kws += 1
        self.rejected_batches.append(
            {
                "batch_index": batch.batch_index,
                "start_time": round(batch.start_time, 3),
                "end_time": round(batch.end_time, 3),
                "duration": round(batch.duration, 3),
                "kws_confidence": kws_result.confidence,
                "kws_reason": kws_result.reason,
            }
        )
        return False

    def process_chunk(self, audio_chunk: np.ndarray) -> None:
        """
        Processa um bloco contínuo de áudio recebido em streaming.
        
        Args:
            audio_chunk: Array numpy 1D com áudio capturado (ex: a cada 32ms - 100ms).
        """
        # 1. Filtro passa-banda leve na CPU
        if self.preprocessor is not None:
            clean_chunk = self.preprocessor.process(
                audio_chunk,
                apply_noise_reduction=self.apply_noise_filter,
            )
        else:
            clean_chunk = audio_chunk

        # 2. VAD: empacota quando detecta pausa/silêncio
        batches = self.segmenter.push_audio(clean_chunk)
        for batch in batches:
            # 3. Keyword Spotting na CPU: só despacha para GPU se houver wake word (ou modo contínuo)
            self.dispatch_batch_if_activated(batch)

    def flush(self) -> None:
        """
        Força a finalização de qualquer fala que esteja retida no buffer do VAD.
        """
        last_batch = self.segmenter.flush()
        if last_batch:
            self.dispatch_batch_if_activated(last_batch)

    def process_stream(
        self,
        stream_generator: Generator[np.ndarray, None, None],
    ) -> List[dict]:
        """
        Consome um gerador de chunks de áudio contínuo até o final e retorna todos os resultados.
        """
        self.start()
        try:
            for chunk in stream_generator:
                self.process_chunk(chunk)
            self.flush()
        finally:
            self.stop()

        return self.results


def simulate_audio_stream(
    file_path: str,
    target_sample_rate: int = 16000,
    chunk_size_samples: int = 1600,  # 100ms a 16kHz
    simulate_realtime: bool = False,
) -> Generator[np.ndarray, None, None]:
    """
    Simula uma fonte contínua de áudio (P5) lendo um arquivo e gerando pedaços sequenciais.
    Garante reamostragem a 16.000 Hz mono para compatibilidade com Silero VAD e Whisper.
    """
    data, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)

    total_samples = len(data)
    for i in range(0, total_samples, chunk_size_samples):
        chunk = data[i : i + chunk_size_samples]
        if simulate_realtime:
            # Pausa para simular o tempo real de captura do microfone
            chunk_duration = len(chunk) / target_sample_rate
            time.sleep(chunk_duration)
        yield chunk


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline I5: VAD Contínuo + Transcrição Whisper em Tempo Real",
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "audio6.ogg"),
        help="Caminho do arquivo de áudio para teste (default: data/audio6.ogg)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        help="Modelo Whisper a ser carregado (default: openai/whisper-small)",
    )
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        help="Simula taxa de transmissão em tempo real inserindo pausas proporcionais",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Tamanho do chunk de streaming em milissegundos (default: 100ms)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limiar do VAD para ativação de fala (default: 0.5)",
    )
    parser.add_argument(
        "--silence-threshold-ms",
        type=int,
        default=700,
        help="Limiar de silêncio para encerramento de batch (default: 700ms)",
    )
    parser.add_argument(
        "--no-noise-filter",
        action="store_true",
        help="Desativa filtro de ruído",
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
        help="Ativa publicação dos eventos de transcrição no tópico Kafka ods.i5.transcript",
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
        help="Tópico Kafka de destino (default: ods.i5.transcript ou KAFKA_TOPIC)",
    )
    parser.add_argument(
        "--wake-word",
        action="store_true",
        help="Ativa o filtro leve de Keyword Spotting (Wake Word) na CPU antes de enviar lotes ao Whisper",
    )
    parser.add_argument(
        "--wake-words",
        type=str,
        default="lexie",
        help="Termo de ativação configurado (default: lexie)",
    )

    args = parser.parse_args()

    audio_file = Path(args.audio_path)
    if not audio_file.exists():
        print(f"Erro: Arquivo '{audio_file}' não encontrado.")
        sys.exit(1)

    kafka_pub = None
    if args.publish_kafka:
        kafka_pub = TranscriptKafkaPublisher(
            bootstrap_servers=args.kafka_broker,
            topic=args.kafka_topic,
        )
        print(f"[I5 - Kafka] Publicando em {kafka_pub.bootstrap_servers} -> Tópico: {kafka_pub.topic}")

    wake_words_list = [w.strip() for w in args.wake_words.split(",") if w.strip()]

    print(f"\n=======================================================")
    print(f"   Iniciando Pipeline I5: VAD Contínuo + KWS + Whisper")
    print(f"   Arquivo: {audio_file.name}")
    print(f"   Modo tempo real: {args.simulate_realtime}")
    print(f"   Filtro Wake Word (CPU): {'ATIVO (' + ', '.join(wake_words_list) + ')' if args.wake_word else 'DESATIVADO (Modo Contínuo)'}")
    print(f"=======================================================\n")

    pipeline = RealtimeSpeechPipeline(
        model_name=args.model,
        speech_threshold=args.threshold,
        silence_threshold_ms=args.silence_threshold_ms,
        apply_noise_filter=not args.no_noise_filter,
        origem=args.origem,
        kafka_publisher=kafka_pub,
        enable_wake_word=args.wake_word,
        wake_words=wake_words_list,
    )

    chunk_size = int(16000 * (args.chunk_ms / 1000.0))
    stream = simulate_audio_stream(
        str(audio_file),
        target_sample_rate=16000,
        chunk_size_samples=chunk_size,
        simulate_realtime=args.simulate_realtime,
    )

    start_total = time.perf_counter()
    resultados = pipeline.process_stream(stream)
    elapsed = time.perf_counter() - start_total

    print(f"\n=======================================================")
    print(f"   Processamento Concluído!")
    print(f"   Tempo total: {elapsed:.2f}s")
    print(f"   Total de frases/batches transcritos: {len(resultados)}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()

