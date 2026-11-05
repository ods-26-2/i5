"""
Componente I5 — Transcritor de Áudio Integrado ao Runtime de Aceleração P2 (SIS-1)
e Publicação de Eventos no Barramento Kafka (B2).

Critérios de Aceitação (Integração com P2):
1. A instanciação direta do PyTorch é substituída por chamadas à biblioteca P2
   utilizando a interface fechada: funções `load`, `infer`, `info` e `release`.
2. A função `info(handle)` é chamada para recuperar a versão exata do modelo
   (`engine_version`), e esse valor é inserido no campo `artifact_version`
   do envelope JSON de saída.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import unicodedata
import warnings
from typing import Any, Dict, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import spacy
except ImportError:
    spacy = None

try:
    from src import p2_runtime as default_p2_runtime
    from src.kafka_publisher import (
        DEFAULT_TOPIC,
        TranscriptKafkaPublisher,
        construir_envelope_transcricao,
    )
except ImportError:
    import p2_runtime as default_p2_runtime
    from kafka_publisher import (
        DEFAULT_TOPIC,
        TranscriptKafkaPublisher,
        construir_envelope_transcricao,
    )

warnings.filterwarnings("ignore", category=UserWarning)


class TranscritorWhisper:
    """
    Transcritor de fala do Componente I5 que delega o ciclo de vida e a execução
    do modelo Whisper em GPU exclusivamente ao Runtime de Aceleração P2
    (`load`, `infer`, `info`, `release`).
    """

    def __init__(
        self,
        model_name: str = "whisper-small-trt-fp16:1.0.0",
        device: str = "cuda",
        options: Optional[Dict[str, Any]] = None,
        p2_module: Any = None,
        language: str = "portuguese",
        task: str = "transcribe",
        torch_dtype: str = "float16",
    ):
        """
        Inicializa o transcritor solicitando o carregamento do engine ao Runtime P2
        e consultando `info(handle)` para obter o `artifact_version`.
        """
        self.engine_ref = model_name
        self.device = device
        self.p2 = p2_module or default_p2_runtime
        self.options = dict(options or {})
        if torch_dtype and "torch_dtype" not in self.options:
            self.options["torch_dtype"] = torch_dtype

        # 1. Carrega o engine na GPU via interface fechada P2: load(engine_ref, options)
        self.handle: Optional[str] = self.p2.load(self.engine_ref, self.options)

        # 2. Recupera metadados e versão exata do modelo via P2: info(handle)
        self.engine_info: Dict[str, Any] = self.p2.info(self.handle)
        self.artifact_version: str = str(
            self.engine_info.get("engine_version", self.engine_ref)
        )

        # Configuração de idioma e tarefa
        self.language = language or "portuguese"
        self.task = task or "transcribe"

        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
            except Exception:
                self.nlp = None

    def obter_info_engine(self) -> Dict[str, Any]:
        """Consulta `info(handle)` no P2 e atualiza `artifact_version`."""
        if self.handle is None:
            raise RuntimeError("O handle do engine P2 já foi liberado.")
        self.engine_info = self.p2.info(self.handle)
        self.artifact_version = str(
            self.engine_info.get("engine_version", self.engine_ref)
        )
        return self.engine_info

    def transcrever_audio(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        origem: str = "P5 pendente",
        timestamp_inicio: Optional[str] = None,
        ruido_excessivo: bool = False,
        kafka_publisher: Optional[TranscriptKafkaPublisher] = None,
    ) -> str:
        """
        Executa a transcrição do batch de áudio via `p2.infer(handle, input)`,
        consulta a versão exata em `p2.info(handle)` para preencher `artifact_version`
        e retorna o envelope JSON padrão B1/B2.
        """
        if self.handle is None:
            raise RuntimeError("Engine P2 não está carregado (handle foi liberado).")

        if not timestamp_inicio:
            inicio_dt = datetime.datetime.now(datetime.timezone.utc)
        else:
            inicio_dt = datetime.datetime.fromisoformat(
                timestamp_inicio.replace("Z", "+00:00")
            )

        duracao_segundos = len(audio_array) / sampling_rate
        fim_dt = inicio_dt + datetime.timedelta(seconds=duracao_segundos)

        # Consulta info(handle) para garantir a versão exata do artefato carregado
        info_atual = self.obter_info_engine()
        artifact_version = str(info_atual.get("engine_version", self.artifact_version))

        # Executa a inferência acelerada na GPU via P2: infer(handle, input)
        resultado_infer, metricas_p2 = self.p2.infer(self.handle, audio_array)

        if isinstance(resultado_infer, dict):
            transcricao = str(
                resultado_infer.get("text")
                or resultado_infer.get("texto_transcrito")
                or resultado_infer.get("texto")
                or ""
            ).strip()
            if "confidence" in resultado_infer or "confianca" in resultado_infer:
                confianca = round(
                    float(
                        resultado_infer.get(
                            "confidence", resultado_infer.get("confianca", 0.0)
                        )
                    ),
                    4,
                )
            else:
                confianca = self._calcular_confianca(
                    resultado_infer.get("scores") or resultado_infer.get("token_probs")
                )
        else:
            transcricao = str(resultado_infer).strip()
            confianca = 0.85 if transcricao else 0.0

        transcricao_limpa = self._preprocessar_texto(transcricao)

        extra_payload = {}
        if isinstance(metricas_p2, dict):
            if "latency_ms" in metricas_p2:
                extra_payload["latency_ms"] = metricas_p2["latency_ms"]
            if "gpu_memory_mb" in metricas_p2:
                extra_payload["gpu_memory_mb"] = metricas_p2["gpu_memory_mb"]

        envelope = construir_envelope_transcricao(
            texto_transcrito=transcricao,
            confianca=confianca,
            origem=origem,
            texto_preprocessado=transcricao_limpa,
            timestamp_inicio=inicio_dt.isoformat(),
            timestamp_fim=fim_dt.isoformat(),
            ruido_excessivo=ruido_excessivo,
            schema=kafka_publisher.topic
            if kafka_publisher
            else os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC),
            artifact_version=artifact_version,
            extra_payload=extra_payload if extra_payload else None,
        )

        # Garante explicitamente artifact_version no topo do envelope JSON de saída
        envelope["artifact_version"] = artifact_version

        if kafka_publisher is not None:
            kafka_publisher.publish_envelope(envelope, flush=True)

        return json.dumps(envelope, ensure_ascii=False, indent=2)

    def _calcular_confianca(self, scores: Any) -> float:
        """
        Converte pontuações/probabilidades dos tokens gerados pelo engine em uma
        probabilidade média no intervalo [0.0, 1.0] sem depender de PyTorch.
        """
        if not scores:
            return 0.0

        probabilidades_tokens = []
        for item in scores:
            if isinstance(item, (int, float)):
                probabilidades_tokens.append(float(item))
            elif np is not None:
                arr = np.asarray(item, dtype=np.float64).reshape(-1)
                if arr.size == 0:
                    continue
                if np.all((arr >= 0.0) & (arr <= 1.0)) and np.isclose(np.sum(arr), 1.0, atol=1e-2):
                    prob_maxima = float(np.max(arr))
                else:
                    shifted = arr - np.max(arr)
                    exp_vals = np.exp(shifted)
                    probs = exp_vals / np.sum(exp_vals)
                    prob_maxima = float(np.max(probs))
                probabilidades_tokens.append(prob_maxima)
            elif isinstance(item, (list, tuple)) and item:
                vals = [float(x) for x in item]
                if all(0.0 <= v <= 1.0 for v in vals) and abs(sum(vals) - 1.0) <= 1e-2:
                    prob_maxima = max(vals)
                else:
                    max_v = max(vals)
                    exp_vals = [math.exp(v - max_v) for v in vals]
                    soma = sum(exp_vals)
                    prob_maxima = max(v / soma for v in exp_vals)
                probabilidades_tokens.append(prob_maxima)

        if not probabilidades_tokens:
            return 0.0

        confianca_media = sum(probabilidades_tokens) / len(probabilidades_tokens)
        return round(confianca_media, 4)

    def _preprocessar_texto(self, texto: str) -> str:
        """Aplica limpeza leve no texto transcrito."""
        texto = texto.lower()
        texto = "".join(
            c
            for c in unicodedata.normalize("NFD", texto)
            if unicodedata.category(c) != "Mn"
        )
        texto = re.sub(r"(.)\1{2,}", r"\1", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        if self.nlp and texto:
            doc = self.nlp(texto)
            texto = " ".join([token.lemma_ for token in doc])
        return texto

    def release(self) -> None:
        """
        Libera o modelo da memória da GPU chamando `p2.release(handle)`.
        """
        if self.handle is not None:
            handle_atual = self.handle
            self.handle = None
            self.p2.release(handle_atual)

    def close(self) -> None:
        """Alias para `release()`."""
        self.release()

    def __enter__(self) -> "TranscritorWhisper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcrição Whisper I5 via Runtime P2 + Publicação Kafka (B2)"
    )
    parser.add_argument(
        "--audio", default="../data/audio6.ogg", help="Caminho do arquivo de áudio"
    )
    parser.add_argument(
        "--engine-ref",
        default="whisper-small-trt-fp16:1.0.0",
        help="artifact_version / engine_ref do modelo no P2",
    )
    parser.add_argument(
        "--origem", default="mic_balcao_1", help="Identificador da origem/microfone"
    )
    parser.add_argument(
        "--broker",
        default=None,
        help="Endereço do broker Kafka (default: 34.66.235.91:9093)",
    )
    parser.add_argument(
        "--topic", default=None, help="Tópico Kafka (default: ods.i5.transcript)"
    )
    parser.add_argument(
        "--no-kafka",
        action="store_true",
        help="Desabilita publicação no broker Kafka",
    )
    args = parser.parse_args()

    transcritor = TranscritorWhisper(model_name=args.engine_ref)
    publisher = None
    if not args.no_kafka:
        try:
            publisher = TranscriptKafkaPublisher(
                bootstrap_servers=args.broker,
                topic=args.topic,
            )
            print(
                f"[I5 - Kafka] Produtor inicializado -> Broker: {publisher.bootstrap_servers} | Tópico: {publisher.topic}"
            )
        except Exception as exc:
            print(
                f"[I5 - Kafka] Aviso: não foi possível inicializar o produtor Kafka ({exc}). Salvando apenas em disco."
            )

    caminho_audio = args.audio

    try:
        audio_real, sr = librosa.load(caminho_audio, sr=16000)

        print(f"A processar o ficheiro: {caminho_audio} (Engine P2: {transcritor.artifact_version})...\n")

        json_saida = transcritor.transcrever_audio(
            audio_array=audio_real,
            origem=args.origem,
            timestamp_inicio="2026-09-16T14:05:00.000Z",
            kafka_publisher=publisher,
        )

        print("--- Saída do Componente I5 (Contrato JSON B2 + P2 artifact_version) ---")
        print(json_saida)

        pasta_saida = "../jsons_gerados"
        os.makedirs(pasta_saida, exist_ok=True)

        timestamp_arquivo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nome_arquivo = f"evento_i5_{timestamp_arquivo}.json"
        caminho_arquivo = os.path.join(pasta_saida, nome_arquivo)

        with open(caminho_arquivo, "w", encoding="utf-8") as f:
            f.write(json_saida)

        print(f"\nJSON salvo com sucesso em: {caminho_arquivo}")

    except FileNotFoundError:
        print(f"Erro: O ficheiro '{caminho_audio}' não foi encontrado.")
    finally:
        transcritor.release()
