"""
Módulo de Publicação de Eventos no Barramento Kafka (Integração B2 - Componente I5).

Implementa o produtor Kafka do componente I5 conforme o "Guia de Uso do Kafka —
Transporte de Eventos (B2)" (Squad TRA-1):
- Biblioteca cliente recomendada: `confluent-kafka`
- Broker configurável via variável de ambiente `KAFKA_BOOTSTRAP_SERVERS`
  (padrão: `34.66.235.91:9093`)
- Tópico padrão: `ods.i5.transcript` (configurável via `KAFKA_TOPIC`)
- Envelope padrão B1/B2 serializado em JSON válido UTF-8 contendo:
  `message_id`, `message_type: "event"`, `schema`, `schema_version`,
  `producer: "I5"`, `occurred_at` e `payload` (com a flag booleana `uncertain`).
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from confluent_kafka import Producer as ConfluentProducer
except ImportError:
    ConfluentProducer = None

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_SERVERS = "34.66.235.91:9093"
DEFAULT_TOPIC = "ods.i5.transcript"
DEFAULT_SCHEMA_VERSION = "1.0"
PRODUCER_ID = "I5"

CAMPOS_OBRIGATORIOS_ENVELOPE = [
    "message_id",
    "message_type",
    "schema",
    "schema_version",
    "producer",
    "occurred_at",
    "payload",
]


def utc_now_iso() -> str:
    """Retorna o timestamp UTC atual no formato ISO 8601 terminado em 'Z'."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def avaliar_incerteza(
    confianca: float,
    limiar_confianca: float = 0.5,
    ruido_excessivo: bool = False,
    texto_transcrito: str = "",
) -> bool:
    """
    Avalia se o resultado da transcrição deve receber `uncertain = True`.

    Retorna True sempre que:
    - A confiança matemática for menor que o limiar (ex: < 0.5);
    - Houver indicação explícita de degradação/ruído excessivo no áudio;
    - A transcrição resultar vazia mesmo após detecção de voz pelo VAD.
    """
    if ruido_excessivo:
        return True
    try:
        conf_val = float(confianca)
    except (TypeError, ValueError):
        return True

    if conf_val < limiar_confianca:
        return True

    if not isinstance(texto_transcrito, str) or not texto_transcrito.strip():
        return True

    return False


def construir_envelope_transcricao(
    texto_transcrito: str,
    confianca: float,
    origem: str = "mic_balcao_1",
    texto_preprocessado: Optional[str] = None,
    timestamp_inicio: Optional[str] = None,
    timestamp_fim: Optional[str] = None,
    ruido_excessivo: bool = False,
    limiar_confianca: float = 0.5,
    schema: Optional[str] = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    occurred_at: Optional[str] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Constrói o envelope padrão B1/B2 para publicação de transcrições do componente I5.
    """
    agora_iso = occurred_at or utc_now_iso()
    ts_inicio = timestamp_inicio or agora_iso
    ts_fim = timestamp_fim or agora_iso
    schema_final = schema or os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC)

    confianca_float = round(float(confianca), 4)
    uncertain = avaliar_incerteza(
        confianca=confianca_float,
        limiar_confianca=limiar_confianca,
        ruido_excessivo=ruido_excessivo,
        texto_transcrito=texto_transcrito,
    )

    payload: Dict[str, Any] = {
        "origem": origem,
        "timestamp_inicio": ts_inicio,
        "timestamp_fim": ts_fim,
        "texto_transcrito": texto_transcrito,
        "texto_preprocessado": (
            texto_preprocessado
            if texto_preprocessado is not None
            else texto_transcrito.lower().strip()
        ),
        "confianca_geral": confianca_float,
        "confidence": confianca_float,
        "uncertain": bool(uncertain),
        "duvida_explicita": bool(uncertain),
    }

    if extra_payload:
        payload.update(extra_payload)

    envelope: Dict[str, Any] = {
        "message_id": str(uuid.uuid4()),
        "message_type": "event",
        "schema": schema_final,
        "schema_version": str(schema_version),
        "producer": PRODUCER_ID,
        "occurred_at": agora_iso,
        "published_at": agora_iso,
        "payload": payload,
    }
    return envelope


def validar_envelope_i5(
    evento: Dict[str, Any],
    schema_esperado: Optional[str] = None,
    versao_suportada_prefixo: str = "1.",
) -> Tuple[bool, Optional[str]]:
    """
    Valida o envelope B1/B2 em três níveis conforme a Seção 4.3 do Guia do Kafka:
    1) Campos obrigatórios do envelope;
    2) Tipo de evento (`message_type == "event"`), `producer == "I5"`, `schema` e `schema_version`;
    3) Estrutura do `payload` (incluindo a flag booleana obrigatória `uncertain` e confiança em [0, 1]).
    """
    if not isinstance(evento, dict):
        return False, "evento não é um dicionário JSON"

    for campo in CAMPOS_OBRIGATORIOS_ENVELOPE:
        if campo not in evento:
            return False, f"campo obrigatório ausente: {campo}"

    if evento.get("message_type") != "event":
        return False, f"message_type inválido: {evento.get('message_type')}"

    if evento.get("producer") != PRODUCER_ID:
        return False, f"producer inesperado: {evento.get('producer')}"

    if schema_esperado and evento.get("schema") != schema_esperado:
        return False, f"schema inesperado: {evento.get('schema')}"

    if not str(evento.get("schema_version", "")).startswith(versao_suportada_prefixo):
        return False, f"versão não suportada: {evento.get('schema_version')}"

    payload = evento.get("payload")
    if not isinstance(payload, dict):
        return False, "payload não é um objeto"

    if "uncertain" not in payload or not isinstance(payload["uncertain"], bool):
        return False, "payload sem a flag booleana obrigatória 'uncertain'"

    conf = payload.get("confianca_geral", payload.get("confidence"))
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0.0 <= float(conf) <= 1.0):
        return False, "confianca_geral/confidence ausente ou fora do intervalo 0 a 1"

    return True, None


class TranscriptKafkaPublisher:
    """
    Produtor Kafka para publicação dos eventos de transcrição do Componente I5
    no tópico `ods.i5.transcript`.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: Optional[str] = None,
        limiar_confianca: float = 0.5,
        producer_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
        extra_kafka_config: Optional[Dict[str, Any]] = None,
    ):
        self.bootstrap_servers = (
            bootstrap_servers
            or os.getenv("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS)
        ).strip()
        self.topic = (
            topic
            or os.getenv("KAFKA_TOPIC", DEFAULT_TOPIC)
        ).strip()
        self.limiar_confianca = float(limiar_confianca)

        kafka_config: Dict[str, Any] = {
            "bootstrap.servers": self.bootstrap_servers,
        }
        if extra_kafka_config:
            kafka_config.update(extra_kafka_config)
        self.kafka_config = kafka_config

        factory = producer_factory or ConfluentProducer
        if factory is None:
            raise RuntimeError(
                "A biblioteca 'confluent-kafka' não está instalada. "
                "Instale com: pip install confluent-kafka"
            )
        self._producer = factory(self.kafka_config)

    @staticmethod
    def _delivery_Callback(err: Any, msg: Any) -> None:
        if err is not None:
            logger.error("[I5 - Kafka] Falha na entrega da mensagem: %s", err)
        else:
            logger.info(
                "[I5 - Kafka] Evento entregue em %s [partição %s @ offset %s]",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )

    def serialize_envelope(self, envelope: Dict[str, Any]) -> bytes:
        """
        Valida o envelope padrão e serializa como JSON válido em bytes UTF-8.
        """
        valido, motivo = validar_envelope_i5(envelope, schema_esperado=self.topic)
        if not valido:
            raise ValueError(f"Envelope inválido para publicação no Kafka: {motivo}")
        return json.dumps(envelope, ensure_ascii=False).encode("utf-8")

    def publish_envelope(
        self,
        envelope: Dict[str, Any],
        flush: bool = True,
        flush_timeout: float = 5.0,
    ) -> bytes:
        """
        Serializa e publica um envelope já montado no tópico configurado (`ods.i5.transcript`).
        """
        payload_bytes = self.serialize_envelope(envelope)
        self._producer.produce(
            topic=self.topic,
            value=payload_bytes,
            on_delivery=self._delivery_Callback,
        )
        if hasattr(self._producer, "poll"):
            self._producer.poll(0)
        if flush:
            self._producer.flush(flush_timeout)
        return payload_bytes

    def publish_transcription(
        self,
        texto_transcrito: str,
        confianca: float,
        origem: str = "mic_balcao_1",
        texto_preprocessado: Optional[str] = None,
        timestamp_inicio: Optional[str] = None,
        timestamp_fim: Optional[str] = None,
        ruido_excessivo: bool = False,
        extra_payload: Optional[Dict[str, Any]] = None,
        flush: bool = True,
        flush_timeout: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Monta o envelope padrão B1/B2 com `uncertain` calculado e publica no Kafka.
        """
        envelope = construir_envelope_transcricao(
            texto_transcrito=texto_transcrito,
            confianca=confianca,
            origem=origem,
            texto_preprocessado=texto_preprocessado,
            timestamp_inicio=timestamp_inicio,
            timestamp_fim=timestamp_fim,
            ruido_excessivo=ruido_excessivo,
            limiar_confianca=self.limiar_confianca,
            schema=self.topic,
            extra_payload=extra_payload,
        )
        self.publish_envelope(envelope, flush=flush, flush_timeout=flush_timeout)
        return envelope

    def flush(self, timeout: float = 5.0) -> int:
        """Garante que todas as mensagens no buffer interno foram entregues ao broker."""
        return self._producer.flush(timeout)

