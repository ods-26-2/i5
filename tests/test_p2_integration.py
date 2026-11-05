"""
Testes automatizados dos Critérios de Aceitação para Execução Acelerada em GPU
(Integração do Componente I5 com o Runtime P2 — SIS-1).

Critérios de Aceitação validados:
1. A instanciação direta do PyTorch é substituída por chamadas à biblioteca P2
   utilizando a interface fechada: funções `load`, `infer`, `info` e `release`.
2. A função `info(handle)` é chamada para recuperar a versão exata do modelo
   (`engine_version`), e esse valor é inserido no campo `artifact_version`
   do envelope JSON de saída.
"""

import ast
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None


def _make_audio(length: int, value: float = 0.1) -> Any:
    if np is not None:
        return np.ones(length, dtype=np.float32) * value
    return [value] * length

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import p2_runtime
from src.TranscriptionWhisper import TranscritorWhisper
from src.p2_runtime import (
    ENGINE_NOT_FOUND,
    INCOMPATIBLE_VERSION,
    INVALID_INPUT_SHAPE,
    OUT_OF_MEMORY,
    P2RuntimeError,
    RUNTIME_ERROR,
)


class SpyP2Module:
    """
    Dublê de teste que registra todas as chamadas à interface fechada do P2:
    `load`, `info`, `infer` e `release`.
    """

    def __init__(self, engine_version: str = "whisper-small-trt-fp16:2.4.1"):
        self.engine_version = engine_version
        self.calls: List[Tuple[str, Any]] = []
        self.active_handle: Optional[str] = None

    def load(self, engine_ref: str, options: Optional[Dict[str, Any]] = None) -> str:
        handle = "handle_jetson_trt_001"
        self.active_handle = handle
        self.calls.append(("load", {"engine_ref": engine_ref, "options": options}))
        return handle

    def info(self, handle: str) -> Dict[str, Any]:
        self.calls.append(("info", {"handle": handle}))
        return {
            "engine_version": self.engine_version,
            "input_shape": [1, 480000],
            "output_shape": [1, 448],
            "precision": "FP16",
            "max_batch": 1,
        }

    def infer(self, handle: str, input_data: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
        self.calls.append(("infer", {"handle": handle, "shape": getattr(input_data, "shape", None)}))
        output = {
            "text": "Quero dois X-Bacon e um suco de laranja",
            "confidence": 0.94,
        }
        metrics = {
            "latency_ms": 18.4,
            "gpu_memory_mb": 480.0,
        }
        return output, metrics

    def release(self, handle: str) -> None:
        self.calls.append(("release", {"handle": handle}))
        if self.active_handle == handle:
            self.active_handle = None


class TestP2GPUAccelerationAcceptance(unittest.TestCase):
    """Suíte de testes dos critérios de aceitação da integração I5 <-> P2."""

    def test_criterio_1_sem_instanciacao_direta_pytorch_e_ciclo_fechado_p2(self):
        """
        Critério 1: A instanciação direta do PyTorch é substituída por chamadas
        à biblioteca P2 utilizando a interface fechada: `load`, `infer`, `info` e `release`.
        """
        # 1. Verifica na AST de TranscriptionWhisper.py que não há `import torch`
        # nem `WhisperForConditionalGeneration.from_pretrained`
        whisper_file = PROJECT_ROOT / "src" / "TranscriptionWhisper.py"
        source = whisper_file.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)

        self.assertNotIn("torch", imported_modules, "TranscriptionWhisper.py não deve importar torch diretamente.")
        self.assertNotIn("transformers", imported_modules, "TranscriptionWhisper.py não deve instanciar modelos transformers diretamente.")

        # 2. Verifica a execução completa do ciclo load -> info -> infer -> release
        spy_p2 = SpyP2Module(engine_version="whisper-small-orin-fp16:1.3.0")
        audio_dummy = _make_audio(16000 * 2, 0.1)

        with TranscritorWhisper(
            model_name="whisper-small-orin-fp16:1.3.0",
            p2_module=spy_p2,
        ) as transcritor:
            saida_json = transcritor.transcrever_audio(
                audio_array=audio_dummy,
                sampling_rate=16000,
                origem="mic_balcao_1",
            )

        nomes_chamadas = [c[0] for c in spy_p2.calls]
        self.assertIn("load", nomes_chamadas)
        self.assertIn("info", nomes_chamadas)
        self.assertIn("infer", nomes_chamadas)
        self.assertIn("release", nomes_chamadas)
        self.assertEqual(nomes_chamadas[0], "load")
        self.assertEqual(nomes_chamadas[-1], "release")
        self.assertIsNone(spy_p2.active_handle)

        envelope = json.loads(saida_json)
        self.assertEqual(
            envelope["payload"]["texto_transcrito"],
            "Quero dois X-Bacon e um suco de laranja",
        )

    def test_criterio_2_info_handle_preenche_artifact_version_no_envelope_json(self):
        """
        Critério 2: A função `info(handle)` é chamada para recuperar a versão exata
        do modelo, e esse valor é inserido no campo `artifact_version` do envelope JSON de saída.
        """
        versao_exata_b5 = "whisper-small-jetpack6-trt10-fp16:3.1.2"
        spy_p2 = SpyP2Module(engine_version=versao_exata_b5)
        audio_dummy = _make_audio(16000, 0.2)

        transcritor = TranscritorWhisper(
            model_name="whisper-small",
            p2_module=spy_p2,
        )
        try:
            saida_json = transcritor.transcrever_audio(
                audio_array=audio_dummy,
                sampling_rate=16000,
                origem="mic_totem_01",
            )
        finally:
            transcritor.release()

        envelope = json.loads(saida_json)

        # Verifica que `info` foi chamado com o handle retornado por `load`
        chamadas_info = [c for c in spy_p2.calls if c[0] == "info"]
        self.assertGreaterEqual(len(chamadas_info), 1)
        self.assertEqual(chamadas_info[0][1]["handle"], "handle_jetson_trt_001")

        # Verifica que `artifact_version` no envelope JSON tem exatamente o valor retornado por `info(handle)`
        self.assertIn("artifact_version", envelope)
        self.assertEqual(envelope["artifact_version"], versao_exata_b5)

    def test_contrato_p2_runtime_orcamento_memoria_e_codigos_de_erro(self):
        """
        Valida a especificação da tabela P2 (SIS-1):
        - `load(engine_ref, options)` -> `ENGINE_NOT_FOUND`, `INCOMPATIBLE_VERSION`, `OUT_OF_MEMORY`
        - `info(handle)` -> `{engine_version, input_shape, output_shape, precision, max_batch}`
        - `infer(handle, input)` -> `(output, {latency_ms, gpu_memory_mb})` e `INVALID_INPUT_SHAPE`
        - `release(handle)` -> libera memória do orçamento da Jetson
        """
        # 1. ENGINE_NOT_FOUND
        with self.assertRaises(P2RuntimeError) as ctx_nf:
            p2_runtime.load("")
        self.assertEqual(ctx_nf.exception.code, ENGINE_NOT_FOUND)

        # 2. INCOMPATIBLE_VERSION
        with self.assertRaises(P2RuntimeError) as ctx_inc:
            p2_runtime.load("whisper-small:0.0.0")
        self.assertEqual(ctx_inc.exception.code, INCOMPATIBLE_VERSION)

        # 3. OUT_OF_MEMORY quando ultrapassa o budget da Jetson Orin Nano
        with self.assertRaises(P2RuntimeError) as ctx_oom:
            p2_runtime.load(
                "whisper-large-trt:1.0.0",
                options={"memory_budget_mb": 512.0, "required_memory_mb": 1024.0},
            )
        self.assertEqual(ctx_oom.exception.code, OUT_OF_MEMORY)

        # 4. Ciclo normal com medição de latência, memória e liberação de handle
        handle = p2_runtime.load(
            "whisper-small-trt-fp16:1.2.0",
            options={"memory_budget_mb": 1024.0, "required_memory_mb": 450.0},
        )
        try:
            meta = p2_runtime.info(handle)
            self.assertEqual(meta["engine_version"], "whisper-small-trt-fp16:1.2.0")
            self.assertIn("input_shape", meta)
            self.assertIn("output_shape", meta)
            self.assertEqual(meta["precision"], "FP16")
            self.assertEqual(meta["max_batch"], 1)

            # INVALID_INPUT_SHAPE ao passar array vazio
            with self.assertRaises(P2RuntimeError) as ctx_shape:
                p2_runtime.infer(handle, _make_audio(0))
            self.assertEqual(ctx_shape.exception.code, INVALID_INPUT_SHAPE)

            # Inferência válida retorna (output, {latency_ms, gpu_memory_mb})
            audio = _make_audio(16000, 0.15)
            out, metrics = p2_runtime.infer(handle, audio)
            self.assertIn("text", out)
            self.assertIn("latency_ms", metrics)
            self.assertIn("gpu_memory_mb", metrics)
            self.assertEqual(metrics["gpu_memory_mb"], 450.0)
        finally:
            p2_runtime.release(handle)

        # 5. Após release(handle), usar o handle antigo deve falhar com RUNTIME_ERROR
        with self.assertRaises(P2RuntimeError) as ctx_rel:
            p2_runtime.info(handle)
        self.assertEqual(ctx_rel.exception.code, RUNTIME_ERROR)

    def test_criterio_4_whisper_bridge_oficial_p2(self):
        """
        Valida que a classe WhisperBridge fornecida oficialmente pela equipe P2
        (examples/whisper_example.py) implementa a interface fechada P2 (load, infer,
        info, release) e se integra perfeitamente com TranscritorWhisper.
        """
        from src.whisper_example import WhisperBridge

        # 1. Verifica os métodos obrigatórios do contrato P2
        self.assertTrue(callable(getattr(WhisperBridge, "load", None)))
        self.assertTrue(callable(getattr(WhisperBridge, "infer", None)))
        self.assertTrue(callable(getattr(WhisperBridge, "info", None)))
        self.assertTrue(callable(getattr(WhisperBridge, "release", None)))

        # 2. Testa a injeção do WhisperBridge como p2_module no TranscritorWhisper
        class MockBridge(WhisperBridge):
            def __init__(self):
                self.calls = []
                self._encoder_ref = "whisper-small-encoder-1.0.0"
                self._decoder_ref = "whisper-small-decoder-1.0.0"

            def load(self, engine_ref="", options=None):
                self.calls.append("load")
                return "mock_bridge_handle_123"

            def info(self, handle):
                self.calls.append("info")
                return {
                    "engine_version": "whisper-small-p2-official:1.0.0",
                    "input_shape": {"input_features": [1, 80, 3000]},
                    "output_shape": {"logits": [1, 448]},
                    "precision": "FP16",
                    "max_batch": 1,
                }

            def infer(self, handle, audio):
                self.calls.append("infer")
                return (
                    {
                        "text": "Dois hambúrgueres e uma porção de batata",
                        "confidence": 0.96,
                        "token_probs": [0.96, 0.95, 0.97],
                    },
                    {"latency_ms": 22.5, "gpu_memory_mb": 420.0},
                )

            def release(self, handle):
                self.calls.append("release")

        bridge_inst = MockBridge()
        transcritor = TranscritorWhisper(
            model_name="whisper-small",
            p2_module=bridge_inst,
        )

        audio_dummy = _make_audio(16000 * 2, 0.1)
        try:
            saida = transcritor.transcrever_audio(audio_dummy, sampling_rate=16000)
            envelope = json.loads(saida)
            self.assertEqual(
                envelope["payload"]["texto_transcrito"],
                "Dois hambúrgueres e uma porção de batata",
            )
            self.assertEqual(envelope["artifact_version"], "whisper-small-p2-official:1.0.0")
            self.assertEqual(envelope["payload"]["confidence"], 0.96)
            self.assertEqual(envelope["payload"]["confianca_geral"], 0.96)
        finally:
            transcritor.release()

        self.assertIn("load", bridge_inst.calls)
        self.assertIn("info", bridge_inst.calls)
        self.assertIn("infer", bridge_inst.calls)
        self.assertIn("release", bridge_inst.calls)


if __name__ == "__main__":
    unittest.main()
