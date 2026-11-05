"""
Componente I5 — Keyword Spotting (Termo de Ativação / Wake Word) na CPU.

Objetivo:
Implementar filtro leve na CPU antes do ASR Whisper (GPU/P2), verificando se o
áudio detectado pelo VAD inicia com uma palavra-chave configurada
(ex: "Totem", "Olá", "A7").

Critérios de Aceitação:
1. O pipeline só despacha o lote de áudio para a GPU se o termo de ativação for
   detectado (ou se o modo de ativação contínua estiver desativado por flag).
2. Redução comprovada de chamadas desnecessárias ao Whisper na presença de
   conversas de fundo.
"""

from __future__ import annotations

import difflib
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_WAKE_WORDS: Tuple[str, ...] = ("lexie",)

# Variações fonéticas/ortográficas equivalentes para "Lexie" (comuns em ASR em português)
LEXIE_ALIASES: Tuple[str, ...] = (
    "lexie",
    "lexi",
    "lxi",
    "lexy",
    "leksi",
    "lex",
    "leci",
    "lecce",
    "alexia",
    "alexa",
)


def normalizar_termo(texto: str) -> str:
    """
    Normaliza uma string para comparação de wake word:
    converte para minúsculas, remove acentos e caracteres especiais nas bordas.
    """
    texto_lower = texto.strip().lower()
    sem_acentos = "".join(
        c
        for c in unicodedata.normalize("NFD", texto_lower)
        if unicodedata.category(c) != "Mn"
    )
    sem_pontuacao = re.sub(r"[^\w\s]", " ", sem_acentos)
    return re.sub(r"\s+", " ", sem_pontuacao).strip()


@dataclass
class WakeWordResult:
    """Resultado da avaliação do filtro de Keyword Spotting na CPU."""

    detected: bool
    matched_keyword: Optional[str] = None
    confidence: float = 0.0
    cpu_time_ms: float = 0.0
    reason: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class KeywordSpotter:
    """
    Filtro leve de Keyword Spotting (Wake Word) executado 100% em CPU antes do
    envio de lotes do VAD para o ASR Whisper na GPU.

    Suporta:
    - Verificação acústica leve na janela inicial do lote (`prefix_window_sec`):
      envelope de energia RMS, modulação silábica (ataque de 1 a 4 sílabas da
      wake word no início do áudio), fator de crista e similaridade com templates.
    - Detector customizado/plugável em CPU (ex.: OpenWakeWord, Vosk, ONNX KWS).
    - Verificação de prefixo textual/metadados quando disponível.
    - Contabilização de métricas de redução de chamadas à GPU.
    """

    def __init__(
        self,
        keywords: Optional[Sequence[str]] = None,
        sample_rate: int = 16000,
        enabled: bool = True,
        prefix_window_sec: float = 1.5,
        confidence_threshold: float = 0.55,
        min_rms_energy: float = 0.04,
        custom_detector: Optional[Callable[[Any, int, List[str]], Optional[Tuple[bool, Optional[str], float]]]] = None,
    ):
        """
        Args:
            keywords: Lista de termos de ativação aceitos (default: ["totem", "olá", "a7", ...]).
            sample_rate: Taxa de amostragem do áudio em Hz (default: 16000).
            enabled: Se True, exige detecção do termo de ativação para aprovar o lote.
                     Se False (modo de ativação contínua), aprova todos os lotes sem filtrar.
            prefix_window_sec: Duração máxima do início do áudio (em segundos) analisada na CPU.
            confidence_threshold: Limiar mínimo de confiança [0.0, 1.0] para aprovar a wake word.
            min_rms_energy: Energia RMS mínima na janela inicial para descartar conversas distantes/ruído de fundo.
            custom_detector: Callback opcional `(prefix_audio, sample_rate, keywords) -> (detected, keyword, confidence)`.
        """
        raw_keywords = list(keywords) if keywords else list(DEFAULT_WAKE_WORDS)
        self.keywords: List[str] = [k.strip() for k in raw_keywords if k.strip()]
        self.normalized_keywords: List[str] = [
            normalizar_termo(k) for k in self.keywords if normalizar_termo(k)
        ]
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.prefix_window_sec = prefix_window_sec
        self.confidence_threshold = confidence_threshold
        self.min_rms_energy = min_rms_energy
        self.custom_detector = custom_detector

        # Templates acústicos de envelope (25 frames na janela inicial) por palavra-chave
        self._templates: Dict[str, List[float]] = {}
        self._init_default_templates()

        # Estatísticas de economia de GPU
        self.total_evaluated: int = 0
        self.total_accepted: int = 0
        self.total_rejected: int = 0

    def _init_default_templates(self) -> None:
        """
        Inicializa o perfil de envelope silábico padrão (25 bins normalizados)
        para o termo de ativação "Lexie" ("Le-xie": 2 sílabas fortes no início).
        """
        lexie_envelope = [
            0.20, 0.85, 1.00, 0.75, 0.30, 0.25, 0.80, 0.95, 0.70, 0.35,
            0.15, 0.10, 0.25, 0.40, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20,
            0.15, 0.10, 0.10, 0.05, 0.05,
        ]
        for alias in LEXIE_ALIASES:
            self._templates[alias] = list(lexie_envelope)

    def register_template(self, keyword: str, audio_samples: Any) -> None:
        """
        Registra um template de áudio de referência para uma palavra-chave,
        extraindo seu envelope de energia em CPU.
        """
        norm_kw = normalizar_termo(keyword)
        if not norm_kw:
            return
        if norm_kw not in self.normalized_keywords:
            self.keywords.append(keyword)
            self.normalized_keywords.append(norm_kw)

        prefix = self._slice_prefix(audio_samples)
        env = self._compute_normalized_envelope(prefix, num_bins=25)
        self._templates[norm_kw] = env

    def detect_in_text(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Verifica se uma string de transcrição ou prefixo inicia com a wake word
        configurada ("Lexie" ou variações fonéticas do ASR como "lexi", "lxi", "lex", "leci").
        Inspeciona as primeiras 4 palavras do lote.
        """
        norm_text = normalizar_termo(text)
        if not norm_text:
            return False, None

        tokens = norm_text.split()
        initial_tokens = tokens[:4]

        for raw_kw, norm_kw in zip(self.keywords, self.normalized_keywords):
            if not norm_kw:
                continue

            if norm_kw in LEXIE_ALIASES:
                for tok in initial_tokens:
                    if (
                        tok in LEXIE_ALIASES
                        or tok.startswith("lex")
                        or tok.startswith("lek")
                        or tok.startswith("lxi")
                        or difflib.SequenceMatcher(None, tok, "lexie").ratio() >= 0.75
                        or difflib.SequenceMatcher(None, tok, "lexi").ratio() >= 0.75
                    ):
                        return True, raw_kw
            else:
                if (
                    norm_text == norm_kw
                    or norm_text.startswith(norm_kw + " ")
                    or norm_kw in initial_tokens
                ):
                    return True, raw_kw

        return False, None

    def check_batch(self, batch: Any) -> WakeWordResult:
        """
        Avalia um `AudioBatch` (ou objeto/dicionário equivalente) na CPU e decide
        se ele deve ser despachado para o Whisper na GPU.
        """
        start_t = time.perf_counter()
        self.total_evaluated += 1

        # Critério 1 (parte B): Se o filtro de wake word estiver desativado
        # (modo de ativação contínua), despacha todos os lotes diretamente.
        if not self.enabled:
            self.total_accepted += 1
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return WakeWordResult(
                detected=True,
                matched_keyword=None,
                confidence=1.0,
                cpu_time_ms=round(elapsed_ms, 3),
                reason="continuous_mode_enabled",
            )

        audio = getattr(batch, "audio", batch)
        sample_rate = getattr(batch, "sample_rate", self.sample_rate)
        metadata = getattr(batch, "metadata", {}) or {}

        # 1. Se houver dica de texto/prefixo nos metadados (ex.: testes ou pré-decoder CPU)
        hint_text = (
            metadata.get("wake_word_phrase")
            or metadata.get("transcript_hint")
            or metadata.get("texto")
        )
        if isinstance(hint_text, str) and hint_text.strip():
            matched, kw = self.detect_in_text(hint_text)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            if matched:
                self.total_accepted += 1
                return WakeWordResult(
                    detected=True,
                    matched_keyword=kw,
                    confidence=0.95,
                    cpu_time_ms=round(elapsed_ms, 3),
                    reason="matched_prefix_text",
                )
            else:
                self.total_rejected += 1
                return WakeWordResult(
                    detected=False,
                    matched_keyword=None,
                    confidence=0.10,
                    cpu_time_ms=round(elapsed_ms, 3),
                    reason="rejected_prefix_text_mismatch",
                )

        # 2. Avalia o sinal de áudio na janela inicial (CPU)
        result = self.check_audio(audio, sample_rate=sample_rate)
        if result.detected:
            self.total_accepted += 1
        else:
            self.total_rejected += 1
        return result

    def check_audio(self, audio_samples: Any, sample_rate: Optional[int] = None) -> WakeWordResult:
        """
        Executa a verificação acústica leve na CPU apenas sobre o prefixo inicial do áudio.
        """
        start_t = time.perf_counter()
        sr = sample_rate or self.sample_rate

        if not self.enabled:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return WakeWordResult(
                detected=True,
                matched_keyword=None,
                confidence=1.0,
                cpu_time_ms=round(elapsed_ms, 3),
                reason="continuous_mode_enabled",
            )

        prefix = self._slice_prefix(audio_samples, sample_rate=sr)
        if len(prefix) == 0:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return WakeWordResult(
                detected=False,
                matched_keyword=None,
                confidence=0.0,
                cpu_time_ms=round(elapsed_ms, 3),
                reason="empty_audio",
            )

        # Se houver detector customizado de CPU configurado (ex: decodificador curto de prefixo 1.5s)
        if self.custom_detector is not None:
            custom_res = self.custom_detector(prefix, sr, self.keywords)
            if custom_res is not None:
                detected, kw, conf = custom_res
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                return WakeWordResult(
                    detected=bool(detected and conf >= self.confidence_threshold),
                    matched_keyword=kw if detected else None,
                    confidence=round(float(conf), 4),
                    cpu_time_ms=round(elapsed_ms, 3),
                    reason="prefix_transcription_matched" if detected else "prefix_transcription_rejected",
                )

        # Análise acústica leve em CPU (RMS, Fator de Crista, Modulação Silábica e Templates)
        detected, matched_kw, confidence, reason, feat_metrics = self._detect_acoustic_wake_word(
            prefix, sr
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return WakeWordResult(
            detected=detected,
            matched_keyword=matched_kw,
            confidence=round(confidence, 4),
            cpu_time_ms=round(elapsed_ms, 3),
            reason=reason,
            metrics=feat_metrics,
        )

    def _slice_prefix(self, audio_samples: Any, sample_rate: Optional[int] = None) -> List[float]:
        """Extrai apenas os primeiros `prefix_window_sec` segundos como lista float."""
        sr = sample_rate or self.sample_rate
        max_samples = max(1, int(sr * self.prefix_window_sec))

        if np is not None and isinstance(audio_samples, np.ndarray):
            flat = audio_samples.reshape(-1)[:max_samples]
            return flat.astype(np.float32).tolist()
        if isinstance(audio_samples, (list, tuple)):
            return [float(x) for x in audio_samples[:max_samples]]
        return []

    def _compute_normalized_envelope(self, samples: List[float], num_bins: int = 25) -> List[float]:
        """Divide as amostras em `num_bins` janelas e retorna o envelope RMS normalizado em [0, 1]."""
        if not samples or num_bins <= 0:
            return [0.0] * max(1, num_bins)

        n = len(samples)
        bin_size = max(1, n // num_bins)
        rms_bins: List[float] = []

        for i in range(num_bins):
            start = i * bin_size
            end = n if i == num_bins - 1 else min(n, (i + 1) * bin_size)
            chunk = samples[start:end]
            if not chunk:
                rms_bins.append(0.0)
            else:
                mean_sq = sum(x * x for x in chunk) / len(chunk)
                rms_bins.append(math.sqrt(mean_sq))

        max_rms = max(rms_bins) if rms_bins else 0.0
        if max_rms <= 1e-9:
            return [0.0] * num_bins
        return [r / max_rms for r in rms_bins]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calcula a similaridade de cosseno entre dois vetores na CPU."""
        if len(vec_a) != len(vec_b) or not vec_a:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a <= 1e-9 or norm_b <= 1e-9:
            return 0.0
        return dot / (norm_a * norm_b)

    def _detect_acoustic_wake_word(
        self, prefix: List[float], sample_rate: int
    ) -> Tuple[bool, Optional[str], float, str, Dict[str, Any]]:
        """
        Verifica se a janela inicial do áudio contém a assinatura acústica de um
        termo de ativação (fala próxima ao microfone com ataque silábico claro no início),
        rejeitando conversas de fundo (energia baixa/difusa, ruído estacionário ou início tardio).
        """
        n = len(prefix)
        mean_sq = sum(x * x for x in prefix) / n
        rms = math.sqrt(mean_sq)
        peak = max(abs(x) for x in prefix)

        # 1. Rejeita conversas de fundo distantes (baixa energia RMS no microfone do balcão/totem)
        if rms < self.min_rms_energy or peak < self.min_rms_energy * 1.8:
            return (
                False,
                None,
                round(min(0.3, rms / max(1e-6, self.min_rms_energy) * 0.3), 4),
                "background_low_energy",
                {"rms": round(rms, 4), "peak": round(peak, 4)},
            )

        # 2. Envelope em 25 bins (~60ms cada para janela de 1.5s)
        envelope = self._compute_normalized_envelope(prefix, num_bins=25)
        mean_env = sum(envelope) / len(envelope)
        variance_env = sum((e - mean_env) ** 2 for e in envelope) / len(envelope)
        std_env = math.sqrt(variance_env)
        modulation_index = std_env / max(1e-6, mean_env)
        crest_factor = peak / max(1e-6, rms)

        # 3. Rejeita ruído contínuo/murmúrio de fundo sem modulação silábica clara
        # Conversas de fundo sobrepostas (babble noise) têm envelope quase plano (baixo modulation_index)
        if modulation_index < 0.32 or crest_factor < 1.65:
            return (
                False,
                None,
                0.25,
                "background_unmodulated_chatter",
                {
                    "rms": round(rms, 4),
                    "modulation_index": round(modulation_index, 4),
                    "crest_factor": round(crest_factor, 4),
                },
            )

        # 4. Verifica se o ataque de fala ocorre logo no início do batch (primeiros 40% da janela)
        early_energy = max(envelope[:10]) if len(envelope) >= 10 else max(envelope)
        if early_energy < 0.55:
            return (
                False,
                None,
                0.30,
                "late_speech_onset",
                {"rms": round(rms, 4), "early_energy": round(early_energy, 4)},
            )

        # 5. Conta pulsos silábicos proeminentes no início (1 a 4 sílabas típicas de "Olá", "Totem", "A7")
        syllable_peaks = 0
        for i in range(1, min(14, len(envelope) - 1)):
            if envelope[i] >= 0.55 and envelope[i] >= envelope[i - 1] and envelope[i] >= envelope[i + 1]:
                syllable_peaks += 1

        # 6. Compara com os templates registrados das palavras-chave ativas
        best_kw: Optional[str] = self.keywords[0] if self.keywords else "lexie"
        best_sim = 0.0
        for raw_kw, norm_kw in zip(self.keywords, self.normalized_keywords):
            tpl = self._templates.get(norm_kw)
            if tpl is not None:
                sim = self._cosine_similarity(envelope, tpl)
                if sim > best_sim:
                    best_sim = sim
                    best_kw = raw_kw

        # Exige alta correspondência com o template acústico da wake word (>= 0.92)
        # para não aprovar falas comuns que não iniciam com "Lexie" (ex: "Oi, eu queria...").
        if best_sim < 0.92:
            return (
                False,
                None,
                round(best_sim * 0.6, 4),
                "wake_word_not_detected_in_prefix",
                {
                    "rms": round(rms, 4),
                    "modulation_index": round(modulation_index, 4),
                    "crest_factor": round(crest_factor, 4),
                    "syllable_peaks": syllable_peaks,
                    "template_similarity": round(best_sim, 4),
                },
            )

        # Combina similaridade de envelope + modulação silábica + ataque inicial
        syllabic_bonus = 0.15 if 1 <= syllable_peaks <= 4 else -0.15
        confidence = min(1.0, max(0.0, 0.75 * best_sim + 0.15 * min(1.0, modulation_index) + syllabic_bonus))

        detected = confidence >= self.confidence_threshold and syllable_peaks >= 1
        reason = "acoustic_wake_word_detected" if detected else "acoustic_score_below_threshold"

        return (
            detected,
            best_kw if detected else None,
            confidence,
            reason,
            {
                "rms": round(rms, 4),
                "modulation_index": round(modulation_index, 4),
                "crest_factor": round(crest_factor, 4),
                "syllable_peaks": syllable_peaks,
                "template_similarity": round(best_sim, 4),
            },
        )

    @property
    def reduction_ratio(self) -> float:
        """
        Retorna a fração [0.0, 1.0] de chamadas ao Whisper economizadas pelo KWS na CPU.
        """
        if self.total_evaluated == 0:
            return 0.0
        return round(self.total_rejected / self.total_evaluated, 4)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas consolidadas do filtro de Keyword Spotting."""
        return {
            "enabled": self.enabled,
            "keywords": list(self.keywords),
            "total_evaluated": self.total_evaluated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "gpu_calls_saved": self.total_rejected,
            "reduction_ratio": self.reduction_ratio,
        }

    @staticmethod
    def synthesize_wake_word_audio(
        keyword: str = "lexie",
        duration_sec: float = 2.0,
        sample_rate: int = 16000,
        amplitude: float = 0.35,
    ) -> Any:
        """
        Utilitário para testes: sintetiza um sinal de áudio 16kHz contendo o envelope
        silábico do termo de ativação ("Lexie") no início seguido de fala.
        """
        total_samples = int(duration_sec * sample_rate)
        spotter = KeywordSpotter(sample_rate=sample_rate)
        norm_kw = normalizar_termo(keyword)
        tpl = spotter._templates.get(norm_kw, spotter._templates["lexie"])

        prefix_samples = min(total_samples, int(1.5 * sample_rate))
        bin_size = max(1, prefix_samples // len(tpl))

        samples: List[float] = []
        for idx in range(total_samples):
            t = idx / sample_rate
            if idx < prefix_samples:
                bin_idx = min(len(tpl) - 1, idx // bin_size)
                env = tpl[bin_idx]
            else:
                # Continuação da frase após a wake word ("... quero um hambúrguer")
                env = 0.45 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 4.0 * t))

            carrier = (
                0.65 * math.sin(2.0 * math.pi * 220.0 * t)
                + 0.35 * math.sin(2.0 * math.pi * 440.0 * t)
            )
            samples.append(amplitude * env * carrier)

        if np is not None:
            return np.asarray(samples, dtype=np.float32)
        return samples

    @staticmethod
    def synthesize_background_chatter(
        duration_sec: float = 2.0,
        sample_rate: int = 16000,
        amplitude: float = 0.02,
    ) -> Any:
        """
        Utilitário para testes: sintetiza conversa de fundo difusa / ruído de ambiente
        que o VAD pode capturar mas que NÃO possui termo de ativação.
        """
        total_samples = int(duration_sec * sample_rate)
        samples: List[float] = []
        for idx in range(total_samples):
            t = idx / sample_rate
            # Murmúrio constante de baixa energia e sem ataque silábico de wake word
            val = amplitude * (
                0.4 * math.sin(2.0 * math.pi * 180.0 * t)
                + 0.35 * math.sin(2.0 * math.pi * 310.0 * t + 0.7)
                + 0.25 * math.sin(2.0 * math.pi * 520.0 * t + 1.3)
            )
            samples.append(val)

        if np is not None:
            return np.asarray(samples, dtype=np.float32)
        return samples

