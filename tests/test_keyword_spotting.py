"""
Testes automatizados dos Critérios de Aceitação para Keyword Spotting
(Termo de Ativação / Wake Word na CPU antes do ASR Whisper).

Critérios de Aceitação validados:
1. O pipeline só despacha o lote de áudio para a GPU se o termo de ativação for
   detectado (ou se o modo de ativação contínua estiver desativado por flag).
2. Redução comprovada de chamadas desnecessárias ao Whisper na presença de
   conversas de fundo.
"""

import json
import sys
import unittest
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.keyword_spotter import KeywordSpotter, normalizar_termo
from src.pipeline import AudioBatch, RealtimeSpeechPipeline


class DummyPreprocessor:
    """Pass-through preprocessor para testes isolados de pipeline."""

    def process(self, audio_chunk: Any, apply_noise_reduction: bool = True) -> Any:
        return audio_chunk


class DummySegmenter:
    """Segmentador controlado que emite lotes pré-definidos para testar o filtro KWS."""

    def __init__(self, batches_to_emit: List[AudioBatch]):
        self._batches = list(batches_to_emit)
        self._emitted = False

    def push_audio(self, clean_chunk: Any) -> List[AudioBatch]:
        if not self._emitted:
            self._emitted = True
            return self._batches
        return []

    def flush(self) -> None:
        return None


class SpyWhisperTranscritor:
    """
    Espião do TranscritorWhisper (GPU/P2) que contabiliza quantas vezes a GPU
    foi acionada pelo pipeline.
    """

    def __init__(self):
        self.gpu_calls: int = 0
        self.transcribed_batches: List[Any] = []
        self.released: bool = False

    def transcrever_audio(
        self,
        audio_array: Any,
        sampling_rate: int = 16000,
        origem: str = "mic_balcao_1",
        **kwargs: Any,
    ) -> str:
        self.gpu_calls += 1
        self.transcribed_batches.append(audio_array)
        return json.dumps(
            {
                "message_id": f"msg-{self.gpu_calls}",
                "message_type": "event",
                "schema": "ods.i5.transcript",
                "schema_version": "1.0",
                "producer": "I5",
                "artifact_version": "whisper-small-trt-fp16:1.0.0",
                "occurred_at": "2026-11-04T22:00:00.000Z",
                "payload": {
                    "origem": origem,
                    "texto_transcrito": "Lexie quero um X-Bacon",
                    "confianca": 0.92,
                    "confianca_geral": 0.92,
                    "uncertain": False,
                },
            }
        )

    def release(self) -> None:
        self.released = True


class TestKeywordSpottingAcceptance(unittest.TestCase):
    """Suíte de testes dos critérios de aceitação de Keyword Spotting (Wake Word: Lexie)."""

    def test_criterio_1_despacha_apenas_com_termo_de_ativacao_ou_modo_continuo_por_flag(self):
        """
        Critério 1: O pipeline só despacha o lote de áudio para a GPU se o termo
        de ativação ("Lexie") for detectado (ou se o modo contínuo estiver ativo por flag).
        """
        audio_wake_lexie_1 = KeywordSpotter.synthesize_wake_word_audio("lexie", duration_sec=2.0)
        audio_wake_lexie_2 = KeywordSpotter.synthesize_wake_word_audio("Lexie", duration_sec=2.0)
        audio_fundo_1 = KeywordSpotter.synthesize_background_chatter(duration_sec=2.0)
        audio_fundo_2 = KeywordSpotter.synthesize_background_chatter(duration_sec=2.0, amplitude=0.015)

        lotes = [
            AudioBatch(batch_index=0, audio=audio_fundo_1, start_time=0.0, end_time=2.0, duration=2.0),
            AudioBatch(batch_index=1, audio=audio_wake_lexie_1, start_time=3.0, end_time=5.0, duration=2.0),
            AudioBatch(batch_index=2, audio=audio_fundo_2, start_time=6.0, end_time=8.0, duration=2.0),
            AudioBatch(batch_index=3, audio=audio_wake_lexie_2, start_time=9.0, end_time=11.0, duration=2.0),
        ]

        # Caso A: Filtro de Wake Word ATIVADO (enable_wake_word=True -> apenas "lexie")
        spy_gpu_filtrado = SpyWhisperTranscritor()
        pipeline_filtrado = RealtimeSpeechPipeline(
            enable_wake_word=True,
            preprocessor=DummyPreprocessor(),
            segmenter=DummySegmenter(lotes),
            transcritor=spy_gpu_filtrado,
        )

        resultados_filtrados = pipeline_filtrado.process_stream([audio_wake_lexie_1])

        self.assertEqual(pipeline_filtrado.keyword_spotter.keywords, ["lexie"])
        self.assertEqual(spy_gpu_filtrado.gpu_calls, 2, "Apenas os 2 lotes com 'Lexie' devem acionar a GPU.")
        self.assertEqual(pipeline_filtrado.batches_dispatched_to_gpu, 2)
        self.assertEqual(pipeline_filtrado.batches_filtered_by_kws, 2)
        self.assertEqual(len(resultados_filtrados), 2)

        # Caso B: Flag de modo contínuo ATIVADA / Wake Word DESATIVADO (enable_wake_word=False / continuous_mode=True)
        spy_gpu_continuo = SpyWhisperTranscritor()
        pipeline_continuo = RealtimeSpeechPipeline(
            enable_wake_word=False,
            continuous_mode=True,
            preprocessor=DummyPreprocessor(),
            segmenter=DummySegmenter(lotes),
            transcritor=spy_gpu_continuo,
        )

        resultados_continuos = pipeline_continuo.process_stream([audio_wake_lexie_1])

        self.assertEqual(
            spy_gpu_continuo.gpu_calls,
            4,
            "Com filtro desativado por flag (modo contínuo), todos os 4 lotes devem ir para a GPU.",
        )
        self.assertEqual(pipeline_continuo.batches_filtered_by_kws, 0)
        self.assertEqual(len(resultados_continuos), 4)

    def test_criterio_2_reducao_comprovada_de_chamadas_ao_whisper_com_conversas_de_fundo(self):
        """
        Critério 2: Redução comprovada de chamadas desnecessárias ao Whisper na
        presença de conversas de fundo.
        Simula 10 segmentos capturados pelo VAD (8 conversas paralelas de fundo / outros termos
        e 2 pedidos reais iniciados por "Lexie").
        """
        audio_lexie = KeywordSpotter.synthesize_wake_word_audio("Lexie", duration_sec=2.0)
        audio_conversa_fundo = KeywordSpotter.synthesize_background_chatter(duration_sec=2.0)

        cenario_10_lotes = [
            AudioBatch(0, audio_conversa_fundo, 0.0, 2.0, 2.0, metadata={"transcript_hint": "você viu o jogo ontem à noite?"}),
            AudioBatch(1, audio_conversa_fundo, 2.5, 4.5, 2.0, metadata={"transcript_hint": "acho que vai chover mais tarde"}),
            AudioBatch(2, audio_lexie, 5.0, 7.0, 2.0, metadata={"transcript_hint": "Lexie quero um combo duplo"}),
            AudioBatch(3, audio_conversa_fundo, 7.5, 9.5, 2.0, metadata={"transcript_hint": "passa o guardanapo por favor"}),
            AudioBatch(4, audio_conversa_fundo, 10.0, 12.0, 2.0, metadata={"transcript_hint": "Totem quero um suco"}),
            AudioBatch(5, audio_conversa_fundo, 12.5, 14.5, 2.0),
            AudioBatch(6, audio_conversa_fundo, 15.0, 17.0, 2.0),
            AudioBatch(7, audio_lexie, 17.5, 19.5, 2.0, metadata={"transcript_hint": "Lexie gostaria de pedir um suco de laranja"}),
            AudioBatch(8, audio_conversa_fundo, 20.0, 22.0, 2.0),
            AudioBatch(9, audio_conversa_fundo, 22.5, 24.5, 2.0),
        ]

        # 1. Sem Keyword Spotting (10 chamadas à GPU)
        spy_sem_kws = SpyWhisperTranscritor()
        pipe_sem_kws = RealtimeSpeechPipeline(
            enable_wake_word=False,
            preprocessor=DummyPreprocessor(),
            segmenter=DummySegmenter(cenario_10_lotes),
            transcritor=spy_sem_kws,
        )
        pipe_sem_kws.process_stream([audio_lexie])

        # 2. Com Keyword Spotting ativo exclusivamente para "Lexie" (apenas 2 chamadas à GPU)
        spy_com_kws = SpyWhisperTranscritor()
        pipe_com_kws = RealtimeSpeechPipeline(
            enable_wake_word=True,
            preprocessor=DummyPreprocessor(),
            segmenter=DummySegmenter(cenario_10_lotes),
            transcritor=spy_com_kws,
        )
        pipe_com_kws.process_stream([audio_lexie])

        self.assertEqual(spy_sem_kws.gpu_calls, 10)
        self.assertEqual(spy_com_kws.gpu_calls, 2)

        stats = pipe_com_kws.keyword_spotter.get_stats()
        self.assertEqual(stats["total_evaluated"], 10)
        self.assertEqual(stats["total_accepted"], 2)
        self.assertEqual(stats["total_rejected"], 8)
        self.assertEqual(stats["gpu_calls_saved"], 8)
        self.assertAlmostEqual(stats["reduction_ratio"], 0.80, places=2)

    def test_deteccao_exclusiva_termo_lexie_na_cpu(self):
        """
        Verifica que apenas o termo "Lexie" (e variações fonéticas diretas "lexi"/"lxi")
        ativa o filtro padrão, rejeitando outros termos ("Totem", "A7", conversas aleatórias).
        """
        spotter = KeywordSpotter(enabled=True)
        self.assertEqual(spotter.keywords, ["lexie"])

        self.assertEqual(spotter.detect_in_text("Lexie, quero dois hambúrgueres"), (True, "lexie"))
        self.assertEqual(spotter.detect_in_text("Oi Lexie quero uma coca"), (True, "lexie"))
        self.assertEqual(spotter.detect_in_text("lxi adicionar batata frita"), (True, "lexie"))
        self.assertEqual(spotter.detect_in_text("Totem quero um lanche"), (False, None))
        self.assertEqual(spotter.detect_in_text("A7 quero um lanche"), (False, None))
        self.assertEqual(spotter.detect_in_text("conversa aleatória na fila"), (False, None))

        # Verificação acústica e tempo de CPU
        sinal_lexie = KeywordSpotter.synthesize_wake_word_audio("lexie", duration_sec=1.8)
        res = spotter.check_audio(sinal_lexie)
        self.assertTrue(res.detected)
        self.assertEqual(res.matched_keyword, "lexie")
        self.assertLess(res.cpu_time_ms, 25.0, "O filtro KWS na CPU deve ser ultraleve (< 25ms).")


if __name__ == "__main__":
    unittest.main()

