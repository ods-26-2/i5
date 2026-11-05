#!/usr/bin/env python3
"""
Pipeline de Streaming e Transcrição em Tempo Real (Componente I5).

Orquestra o fluxo contínuo de áudio configurável via YAML e CLI:
1. Ingestão de áudio em streaming (P5 Mock / Microfone) a 16.000 Hz.
2. Pré-processamento e filtragem passa-banda (AudioPreprocessor na CPU).
3. Detecção de atividade de voz e segmentação contínua (VADSegmenter na CPU).
4. Fila desacoplada em memória RAM (Queue Produtor-Consumidor).
5. Transcrição forçada em Português com timestamps preservados (Whisper na GPU).
6. Emissão de payloads estruturados para A6/A7 (via callback, terminal e JSON).
"""

import argparse
import datetime
import json
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

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

from src.config_loader import load_pipeline_config
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
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
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
        **overrides,
    ):
        """
        Inicializa o pipeline com os componentes de VAD, Keyword Spotting (CPU),
        Transcrição (GPU/P2) e Publicação Kafka (B2), aceitando arquivo YAML e overrides.
        """
        # Carrega configuração base
        if config is not None:
            self.config = config
        else:
            self.config = load_pipeline_config(config_path)

        # Aplica overrides pontuais (ex: passados diretamente via código ou CLI)
        if "sample_rate" in overrides and overrides["sample_rate"] is not None:
            self.config["audio"]["sample_rate"] = overrides["sample_rate"]
        if "speech_threshold" in overrides and overrides["speech_threshold"] is not None:
            self.config["vad"]["speech_threshold"] = overrides["speech_threshold"]
        if "silence_threshold_ms" in overrides and overrides["silence_threshold_ms"] is not None:
            self.config["vad"]["silence_threshold_ms"] = overrides["silence_threshold_ms"]
        if "apply_noise_filter" in overrides and overrides["apply_noise_filter"] is not None:
            self.config["preprocessor"]["apply_noise_filter"] = overrides["apply_noise_filter"]
        if "model_name" in overrides and overrides["model_name"] is not None:
            self.config["whisper"]["model_name"] = overrides["model_name"]
        if "device" in overrides and overrides["device"] is not None:
            self.config["whisper"]["device"] = overrides["device"]
        if "language" in overrides and overrides["language"] is not None:
            self.config["whisper"]["language"] = overrides["language"]

        self.on_transcription = on_transcription
        self.origem = origem
        self.kafka_publisher = kafka_publisher

        audio_cfg = self.config.get("audio", {})
        prep_cfg = self.config.get("preprocessor", {})
        vad_cfg = self.config.get("vad", {})
        whisper_cfg = self.config.get("whisper", {})
        output_cfg = self.config.get("output", {})

        self.sample_rate = audio_cfg.get("sample_rate", 16000)
        self.apply_noise_filter = prep_cfg.get("apply_noise_filter", True)

        # Configurações de saída
        self.save_json = output_cfg.get("save_json", True)
        self.json_dir = output_cfg.get("json_dir", "jsons_gerados")
        self.min_confidence_warning = output_cfg.get("min_confidence_warning", 0.50)

        # Se continuous_mode for informado explicitamente:
        if continuous_mode is not None:
            self.enable_wake_word = not bool(continuous_mode)
            self.continuous_mode = bool(continuous_mode)
        elif keyword_spotter is not None:
            self.enable_wake_word = bool(keyword_spotter.enabled)
            self.continuous_mode = not self.enable_wake_word
        else:
            self.enable_wake_word = bool(enable_wake_word)
            self.continuous_mode = not self.enable_wake_word

        self.device = whisper_cfg.get("device", "auto")
        if self.device == "auto" or self.device is None:
            self.device = "cuda"

        config_source = self.config.get("_config_file") or "Defaults embutidos"
        print(f"[I5 - Pipeline] Configuração carregada de: {config_source}")
        print(f"[I5 - Pipeline] Runtime de inferência P2 (device alvo: {self.device})")

        # Front-End de Áudio (CPU)
        if preprocessor is not None:
            self.preprocessor = preprocessor
        elif AudioPreprocessor is not None:
            self.preprocessor = AudioPreprocessor(
                target_sample_rate=self.sample_rate,
                lowcut=prep_cfg.get("lowcut_hz", 80.0),
                highcut=prep_cfg.get("highcut_hz", 7500.0),
                filter_order=prep_cfg.get("filter_order", 5),
                enable_spectral_subtraction=prep_cfg.get("enable_spectral_subtraction", False),
            )
        else:
            self.preprocessor = None

        # Voice Activity Detection (Silero VAD na CPU via ONNX)
        if segmenter is not None:
            self.segmenter = segmenter
        elif VADSegmenter is not None:
            self.segmenter = VADSegmenter(
                sample_rate=self.sample_rate,
                speech_threshold=vad_cfg.get("speech_threshold", 0.50),
                silence_threshold_ms=vad_cfg.get("silence_threshold_ms", 700),
                pre_speech_pad_ms=vad_cfg.get("pre_speech_pad_ms", 250),
                post_speech_pad_ms=vad_cfg.get("post_speech_pad_ms", 200),
                min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 250),
                use_onnx=vad_cfg.get("use_onnx", True),
            )
        else:
            self.segmenter = None

        # Transcritor ASR (Whisper na GPU via P2)
        if transcritor is not None:
            self.transcritor = transcritor
        else:
            self.transcritor = TranscritorWhisper(
                model_name=whisper_cfg.get("model_name", "openai/whisper-small"),
                device=self.device,
                language=whisper_cfg.get("language", "portuguese"),
                task=whisper_cfg.get("task", "transcribe"),
                torch_dtype=whisper_cfg.get("torch_dtype", "float16"),
            )
        print(f"[I5 - Pipeline] Whisper iniciado no device: {getattr(self.transcritor, 'device', self.device)} (modelo: {whisper_cfg.get('model_name')})")

        # Filtro leve de Keyword Spotting (Wake Word) antes do ASR completo:
        # Quando estamos usando o TranscritorWhisper real (sem custom_detector de teste),
        # podemos verificar apenas a janela inicial de 1.8s transcrevendo somente as primeiras
        # palavras para confirmar se o áudio começa com "Lexie".
        if keyword_spotter is not None:
            self.keyword_spotter = keyword_spotter
            self.keyword_spotter.enabled = self.enable_wake_word
        else:
            prefix_detector = None
            if transcritor is None and hasattr(self.transcritor, "transcrever_audio"):
                def _verificar_prefixo_curto(prefix_samples, sr, keywords):
                    # Se o sinal já tiver similaridade acústica exata com o template sintético (>= 0.92), aceita direto
                    samples_1_5s = prefix_samples[: int(1.5 * sr)]
                    env = self.keyword_spotter._compute_normalized_envelope(samples_1_5s, num_bins=25)
                    for raw_kw, norm_kw in zip(self.keyword_spotter.keywords, self.keyword_spotter.normalized_keywords):
                        tpl = self.keyword_spotter._templates.get(norm_kw)
                        if tpl and self.keyword_spotter._cosine_similarity(env, tpl) >= 0.92:
                            return True, raw_kw, 0.95

                    # Para áudio humano real: transcreve a janela inicial (~3.0s) sem cortar a sílaba
                    # de "Lexie" / "Lexi" ao meio, verificando se o termo de ativação foi dito no início.
                    arr = np.asarray(prefix_samples, dtype=np.float32) if np is not None else prefix_samples
                    res_json = self.transcritor.transcrever_audio(arr, sampling_rate=sr, origem=self.origem)
                    env_dict = json.loads(res_json) if isinstance(res_json, str) else res_json
                    texto_prefixo = env_dict.get("payload", {}).get("texto_transcrito", "")
                    matched, kw = self.keyword_spotter.detect_in_text(texto_prefixo)
                    print(f"[I5 - KWS] Início do áudio ouvido: \"{texto_prefixo}\" -> Wake word ('lexie'/'lexi') detectada: {matched}")
                    return matched, kw, (0.95 if matched else 0.10)

                prefix_detector = _verificar_prefixo_curto

            self.keyword_spotter = KeywordSpotter(
                keywords=wake_words or list(DEFAULT_WAKE_WORDS),
                sample_rate=self.sample_rate,
                enabled=self.enable_wake_word,
                prefix_window_sec=3.0 if prefix_detector is not None else 1.5,
                custom_detector=prefix_detector,
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
        if self.save_json:
            caminho_pasta = PROJECT_ROOT / self.json_dir
            caminho_pasta.mkdir(parents=True, exist_ok=True)
        else:
            caminho_pasta = None

        while self._running or not self.batch_queue.empty():
            try:
                item = self.batch_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                self.batch_queue.task_done()
                break

            batch: AudioBatch = item
            inference_start = time.perf_counter()

            # 1. Base temporal da transcrição ancorada no captured_at de P5
            timestamp_inicio = getattr(batch, "start_at", None) or datetime.datetime.now(datetime.timezone.utc).isoformat()
            origem = batch.metadata.get("origem", self.origem)

            # 2. Chama a IA que retorna o JSON oficial (String)
            json_contrato_string = self.transcritor.transcrever_audio(
                audio_array=batch.audio,
                sampling_rate=batch.sample_rate,
                origem=origem,
                timestamp_inicio=timestamp_inicio,
            )

            inference_time_ms = (time.perf_counter() - inference_start) * 1000

            # 3. Lê o JSON de volta para extrair dados
            if isinstance(json_contrato_string, str):
                dados_contrato = json.loads(json_contrato_string)
            else:
                dados_contrato = json_contrato_string
                json_contrato_string = json.dumps(dados_contrato, ensure_ascii=False)

            payload_interno = dados_contrato.get("payload", {})
            texto_extraido = payload_interno.get("texto_transcrito", "")
            confianca_extraida = payload_interno.get("confianca_geral", payload_interno.get("confidence", 0.0))
            uncertain_flag = payload_interno.get("uncertain", bool(confianca_extraida < 0.5))

            # Publica no barramento Kafka (tópico ods.i5.transcript) se o produtor estiver ativo
            if self.kafka_publisher is not None:
                self.kafka_publisher.publish_envelope(dados_contrato, flush=False)

            self.results.append(dados_contrato)

            # 4. Salva o JSON oficial na pasta configurada
            caminho_arquivo_str = ""
            if self.save_json and caminho_pasta:
                timestamp_arquivo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                caminho_arquivo = caminho_pasta / f"evento_i5_{timestamp_arquivo}.json"
                with open(caminho_arquivo, "w", encoding="utf-8") as f:
                    f.write(json_contrato_string)
                caminho_arquivo_str = f" [Salvo em {caminho_arquivo.relative_to(PROJECT_ROOT)}]"

            # 5. Notificação via callback ou stdout
            if self.on_transcription:
                self.on_transcription(dados_contrato)
            else:
                alerta = " ⚠️ (baixa confiança)" if confianca_extraida < self.min_confidence_warning else ""
                print(
                    f"[{batch.start_time:6.2f}s -> {batch.end_time:6.2f}s] "
                    f"(conf: {confianca_extraida:.2f}{alerta} | uncertain: {uncertain_flag} | {inference_time_ms:.0f}ms) "
                    f"=> \"{texto_extraido}\"{caminho_arquivo_str}"
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
            if self.enable_wake_word:
                print(
                    f"[I5 - KWS (CPU)] Lote #{batch.batch_index} ({batch.start_time:.2f}s -> {batch.end_time:.2f}s) "
                    f"APROVADO (wake word '{kws_result.matched_keyword}' detectada, conf={kws_result.confidence:.2f}) -> enviando à GPU"
                )
            self.batch_queue.put(batch)
            return True

        self.batches_filtered_by_kws += 1
        print(
            f"[I5 - KWS (CPU)] Lote #{batch.batch_index} ({batch.start_time:.2f}s -> {batch.end_time:.2f}s) "
            f"DESCARTADO (wake word {self.keyword_spotter.keywords} não detectada no início: {kws_result.reason}) "
            f"— chamada ao Whisper na GPU economizada!"
        )
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

    def process_chunk(
        self,
        audio_chunk: np.ndarray,
        captured_at: Optional[Union[str, datetime.datetime, float, int]] = None,
    ) -> None:
        """
        Processa um bloco contínuo de áudio recebido em streaming (Kafka ou microfone).
        
        Args:
            audio_chunk: Array numpy 1D com áudio capturado (ex: a cada 32ms - 100ms).
            captured_at: Timestamp de captura emitido pelo hardware/P5 (ISO 8601 ou epoch).
        """
        # 1. Filtro passa-banda leve na CPU
        if self.preprocessor is not None:
            clean_chunk = self.preprocessor.process(
                audio_chunk,
                apply_noise_reduction=self.apply_noise_filter,
            )
        else:
            clean_chunk = audio_chunk

        # 2. VAD: empacota quando detecta pausa/silêncio ancorando captured_at
        try:
            batches = self.segmenter.push_audio(clean_chunk, captured_at=captured_at)
        except TypeError:
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
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Caminho para o arquivo YAML de configuração (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Modelo Whisper (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device de inferência: auto, cuda ou cpu (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Idioma forçado do Whisper (sobrescreve o YAML)",
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
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=None,
        help="Tamanho do chunk de streaming em ms (sobrescreve o YAML)",
    )
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        default=None,
        help="Simula taxa de transmissão em tempo real inserindo pausas proporcionais",
    )
    parser.add_argument(
        "--no-noise-filter",
        action="store_true",
        default=False,
        help="Desativa filtro passa-banda de ruído",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        default=False,
        help="Desativa a gravação de arquivos JSON de evento",
    )
    parser.add_argument(
        "--kafka",
        action="store_true",
        help="Consome áudio em streaming diretamente do barramento Kafka (P5 - ods.p5.audio_chunk)",
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
    parser.add_argument(
        "--replay-json",
        type=str,
        default=None,
        help="Reenvia diretamente um arquivo JSON já gerado (ou da pasta jsons_gerados) para o Kafka ods.i5.transcript",
    )

    args = parser.parse_args()

    # 1. Carrega configuração do YAML
    config = load_pipeline_config(args.config)

    # 2. Aplica overrides da CLI (CLI > YAML)
    if args.threshold is not None:
        config["vad"]["speech_threshold"] = args.threshold
    if args.silence_threshold_ms is not None:
        config["vad"]["silence_threshold_ms"] = args.silence_threshold_ms
    if args.model is not None:
        config["whisper"]["model_name"] = args.model
    if args.device is not None:
        config["whisper"]["device"] = args.device
    if args.language is not None:
        config["whisper"]["language"] = args.language
    if args.chunk_ms is not None:
        config["audio"]["chunk_duration_ms"] = args.chunk_ms
    if args.simulate_realtime:
        config["audio"]["simulate_realtime"] = True
    if args.no_noise_filter:
        config["preprocessor"]["apply_noise_filter"] = False
    if args.no_save_json:
        config["output"]["save_json"] = False

    pipeline = RealtimeSpeechPipeline(config=config)

    # 3. Modo KAFKA: escuta do barramento de eventos (P5)
    if args.kafka:
        from src.kafka_audio_consumer import KafkaAudioConsumer

        kafka_cfg = config.get("kafka", {})
        print(f"\n=======================================================")
        print(f"   Iniciando Pipeline I5 via KAFKA (Streaming P5)")
        print(f"   Broker: {kafka_cfg.get('bootstrap_servers')}")
        print(f"   Tópico: {kafka_cfg.get('topic_input')}")
        print(f"   Group ID: {kafka_cfg.get('group_id')}")
        print(f"   VAD Speech Threshold: {config['vad'].get('speech_threshold')}")
        print(f"=======================================================\n")

        pipeline.start()
        consumer = KafkaAudioConsumer.from_config(config=config)
        try:
            consumer.listen_and_feed(pipeline)
        finally:
            pipeline.flush()
            pipeline.stop()
            print(f"\n[I5 - Pipeline] Finalizado. Total de frases/batches transcritos: {len(pipeline.results)}")
        return

    # Modo Replay de JSON já existente no I5
    if args.replay_json:
        caminho_json = Path(args.replay_json)
        if not caminho_json.exists():
            caminho_json = PROJECT_ROOT / "jsons_gerados" / args.replay_json
        if not caminho_json.exists():
            print(f"Erro: Arquivo JSON '{args.replay_json}' não encontrado.")
            sys.exit(1)

        with open(caminho_json, "r", encoding="utf-8") as f:
            dados = json.load(f)

        pub = TranscriptKafkaPublisher(
            bootstrap_servers=args.kafka_broker,
            topic=args.kafka_topic or "ods.i5.transcript",
        )
        print(f"\n[I5 - Replay] Enviando {caminho_json.name} para o Kafka ({pub.bootstrap_servers} -> {pub.topic})...")
        payload_interno = dados.setdefault("payload", {})
        texto = payload_interno.get("texto_transcrito") or payload_interno.get("texto", "")
        conf = payload_interno.get("confianca_geral", payload_interno.get("confidence", 0.95))

        # Garante conformidade com o contrato moderno B1/B2
        if "uncertain" not in payload_interno:
            payload_interno["uncertain"] = bool(payload_interno.get("duvida_explicita", False) or conf < 0.5)
        if "confianca_geral" not in payload_interno:
            payload_interno["confianca_geral"] = float(conf)
        if "confidence" not in payload_interno:
            payload_interno["confidence"] = float(conf)
        if "origem" not in payload_interno:
            payload_interno["origem"] = args.origem or "mic_balcao_1"
        if "timestamp_inicio" not in payload_interno:
            payload_interno["timestamp_inicio"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if "timestamp_fim" not in payload_interno:
            payload_interno["timestamp_fim"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Garante message_id único e timestamp fresco
        import uuid
        dados["message_id"] = str(uuid.uuid4())
        dados["message_type"] = "event"
        dados["producer"] = "I5"
        dados["schema_version"] = "1.0"
        dados["artifact_version"] = dados.get("artifact_version", "whisper-small-trt-fp16:1.0.0")
        dados["published_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        dados["occurred_at"] = dados["published_at"]
        dados["schema"] = pub.topic

        pub.publish_envelope(dados, flush=True)
        print(f"[I5 - Replay] Sucesso! Texto: \"{texto}\" (conf: {conf})\n")
        return

    # 4. Modo Local/Arquivo
    audio_file = Path(args.audio_path)
    if not audio_file.exists():
        print(f"Erro: Arquivo '{audio_file}' não encontrado.")
        sys.exit(1)

    sample_rate = config["audio"].get("sample_rate", 16000)
    chunk_ms = config["audio"].get("chunk_duration_ms", 100)
    simulate_realtime = config["audio"].get("simulate_realtime", False)

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
    print(f"   Modo tempo real: {simulate_realtime}")
    print(f"   Filtro Wake Word (CPU): {'ATIVO (' + ', '.join(wake_words_list) + ')' if args.wake_word else 'DESATIVADO (Modo Contínuo)'}")
    print(f"   VAD Speech Threshold: {config['vad'].get('speech_threshold')}")
    print(f"   VAD Silence Threshold: {config['vad'].get('silence_threshold_ms')}ms")
    print(f"=======================================================\n")

    pipeline = RealtimeSpeechPipeline(
        config=config,
        origem=args.origem,
        kafka_publisher=kafka_pub,
        enable_wake_word=args.wake_word,
        wake_words=wake_words_list,
    )

    chunk_size = int(sample_rate * (chunk_ms / 1000.0))
    stream = simulate_audio_stream(
        str(audio_file),
        target_sample_rate=sample_rate,
        chunk_size_samples=chunk_size,
        simulate_realtime=simulate_realtime,
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
