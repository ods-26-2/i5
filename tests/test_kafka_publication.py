"""
Testes automatizados dos Critérios de Aceitação para Publicação de Transcrições
no Barramento Kafka (Integração B2 - Componente I5).

Critérios de Aceitação validados:
1. O I5 cria um produtor Kafka apontado para o broker (padrão 34.66.235.91:9093,
   configurável) e publica eventos no tópico `ods.i5.transcript`.
2. A mensagem é serializada como um JSON válido UTF-8 contendo o envelope padrão:
   `message_id`, `message_type: "event"`, `schema`, `schema_version`,
   `producer: "I5"` e `occurred_at`.
3. O payload inclui obrigatoriamente a flag booleana `uncertain = True` sempre que
   as condições de ruído ou confiança matemática o exigirem (e `False` caso contrário).
"""

import json
import os
import sys
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kafka_publisher import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_TOPIC,
    TranscriptKafkaPublisher,
    construir_envelope_transcricao,
    validar_envelope_i5,
)


class FakeKafkaProducer:
    """Mock de confluent_kafka.Producer para validar chamadas sem tráfego em nuvem."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.produced_messages: List[Dict[str, Any]] = []
        self.flush_calls: int = 0

    def produce(self, topic: str, value: bytes, on_delivery: Any = None) -> None:
        self.produced_messages.append(
            {
                "topic": topic,
                "value": value,
                "on_delivery": on_delivery,
            }
        )

    def poll(self, timeout: float = 0) -> int:
        return 0

    def flush(self, timeout: float = 5.0) -> int:
        self.flush_calls += 1
        return 0


class TestKafkaPublicationAcceptance(unittest.TestCase):
    """Suíte de testes dos critérios de aceitação da integração B2 (Kafka) no I5."""

    def test_criterio_1_produtor_apontado_para_broker_e_topico_ods_i5_transcript(self):
        """
        Critério 1: O I5 cria um produtor Kafka apontado para o broker e publica
        eventos no tópico `ods.i5.transcript`.
        """
        created_producers: List[FakeKafkaProducer] = []

        def factory(cfg: Dict[str, Any]) -> FakeKafkaProducer:
            prod = FakeKafkaProducer(cfg)
            created_producers.append(prod)
            return prod

        publisher = TranscriptKafkaPublisher(producer_factory=factory)

        self.assertEqual(len(created_producers), 1)
        fake_prod = created_producers[0]
        self.assertEqual(
            fake_prod.config.get("bootstrap.servers"),
            DEFAULT_BOOTSTRAP_SERVERS,
        )
        self.assertEqual(publisher.topic, "ods.i5.transcript")

        publisher.publish_transcription(
            texto_transcrito="quero dois x-saladas",
            confianca=0.92,
            origem="mic_balcao_1",
        )

        self.assertEqual(len(fake_prod.produced_messages), 1)
        msg = fake_prod.produced_messages[0]
        self.assertEqual(msg["topic"], "ods.i5.transcript")
        self.assertIsInstance(msg["value"], bytes)
        self.assertGreaterEqual(fake_prod.flush_calls, 1)

    def test_criterio_1_broker_configuravel_por_variavel_ambiente(self):
        """
        Garante que o endereço do broker pode ser trocado sem alterar o código
        (conforme exigido pelo Guia do Kafka - B2).
        """
        os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "192.168.1.50:9092"
        try:
            captured_cfg = {}

            def factory(cfg: Dict[str, Any]) -> FakeKafkaProducer:
                captured_cfg.update(cfg)
                return FakeKafkaProducer(cfg)

            pub = TranscriptKafkaPublisher(producer_factory=factory)
            self.assertEqual(pub.bootstrap_servers, "192.168.1.50:9092")
            self.assertEqual(captured_cfg.get("bootstrap.servers"), "192.168.1.50:9092")
        finally:
            os.environ.pop("KAFKA_BOOTSTRAP_SERVERS", None)

    def test_criterio_2_serializacao_json_utf8_e_envelope_padrao(self):
        """
        Critério 2: A mensagem é serializada como um JSON válido UTF-8 contendo
        o envelope padrão: `message_id`, `message_type: "event"`, `schema`,
        `schema_version`, `producer: "I5"` e `occurred_at`.
        """
        fake_prod = None

        def factory(cfg: Dict[str, Any]) -> FakeKafkaProducer:
            nonlocal fake_prod
            fake_prod = FakeKafkaProducer(cfg)
            return fake_prod

        publisher = TranscriptKafkaPublisher(producer_factory=factory)
        publisher.publish_transcription(
            texto_transcrito="Gostaria de um açaí e pão de queijo com café!",
            confianca=0.88,
            origem="mic_mesa_05",
        )

        raw_bytes = fake_prod.produced_messages[0]["value"]
        self.assertIsInstance(raw_bytes, bytes)

        # Decodifica estritamente em UTF-8 e faz parse JSON
        decoded_str = raw_bytes.decode("utf-8")
        self.assertIn("açaí", decoded_str)
        self.assertIn("pão", decoded_str)
        evento = json.loads(decoded_str)

        # Valida campos obrigatórios do envelope padrão B1/B2
        self.assertIn("message_id", evento)
        uuid.UUID(evento["message_id"])  # Garante UUID válido
        self.assertEqual(evento["message_type"], "event")
        self.assertEqual(evento["schema"], "ods.i5.transcript")
        self.assertEqual(evento["schema_version"], "1.0")
        self.assertEqual(evento["producer"], "I5")
        self.assertIn("occurred_at", evento)
        self.assertTrue(evento["occurred_at"].endswith("Z"))

        valido, motivo = validar_envelope_i5(evento, schema_esperado=DEFAULT_TOPIC)
        self.assertTrue(valido, f"Envelope deveria ser válido, mas falhou: {motivo}")

    def test_criterio_3_flag_booleana_uncertain_por_confianca_ou_ruido(self):
        """
        Critério 3: O payload inclui obrigatoriamente a flag booleana
        `uncertain = True` sempre que as condições de ruído ou confiança
        matemática o exigirem.
        """
        # Caso A: Confiança alta (0.91) e sem ruído excessivo -> uncertain = False
        env_alta = construir_envelope_transcricao(
            texto_transcrito="quero uma coca cola",
            confianca=0.91,
            ruido_excessivo=False,
        )
        self.assertIs(env_alta["payload"]["uncertain"], False)

        # Caso B: Confiança matemática baixa (0.35 < 0.50) -> uncertain = True
        env_baixa_conf = construir_envelope_transcricao(
            texto_transcrito="hum acho que...",
            confianca=0.35,
            ruido_excessivo=False,
        )
        self.assertIs(env_baixa_conf["payload"]["uncertain"], True)

        # Caso C: Confiança boa (0.78), mas condição de ruído excessivo detectada -> uncertain = True
        env_ruido = construir_envelope_transcricao(
            texto_transcrito="quero um x-bacon",
            confianca=0.78,
            ruido_excessivo=True,
        )
        self.assertIs(env_ruido["payload"]["uncertain"], True)

        # Caso D: Transcrição vazia após VAD -> uncertain = True
        env_vazio = construir_envelope_transcricao(
            texto_transcrito="   ",
            confianca=0.80,
            ruido_excessivo=False,
        )
        self.assertIs(env_vazio["payload"]["uncertain"], True)

    def test_validacao_envelope_rejeita_mensagens_fora_do_contrato(self):
        """
        Verifica a validação em 3 níveis da Seção 4.3 do Guia do Kafka.
        """
        valido, _ = validar_envelope_i5({"message_type": "event"})
        self.assertFalse(valido)

        env_sem_uncertain = construir_envelope_transcricao("teste", 0.9)
        del env_sem_uncertain["payload"]["uncertain"]
        valido2, motivo2 = validar_envelope_i5(env_sem_uncertain)
        self.assertFalse(valido2)
        self.assertIn("uncertain", motivo2)


if __name__ == "__main__":
    unittest.main()
