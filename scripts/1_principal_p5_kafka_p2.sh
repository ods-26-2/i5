#!/usr/bin/env bash
# ==============================================================================
# CENÁRIO 1 (PRINCIPAL): Escuta Ativa de P5 com Kafka e Inferência P2 (Whisper)
# ==============================================================================
# Executa o pipeline completo oficial do I5:
# 1. Consome stream contínuo de áudio da Camada 1 (P5 - ods.p5.audio_chunk) via Kafka.
# 2. Executa pré-processamento (Butterworth) e Silero VAD na CPU.
# 3. Aplica o filtro de Wake Word da Ester ("Lexie") na CPU.
# 4. Transcreve lotes aprovados na GPU usando o runtime acelerado P2 (TensorRT).
# 5. Publica eventos de transcrição canônicos em ods.i5.transcript (para o A7).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
I5_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$I5_DIR"

if [ -f "$I5_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$I5_DIR/.venv/bin/python3"
elif [ -f "$I5_DIR/venv/bin/python3" ]; then
    PYTHON_EXEC="$I5_DIR/venv/bin/python3"
elif [ -f "$I5_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$I5_DIR/venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="$I5_DIR"

echo "=================================================================="
echo "   [CENÁRIO 1 - PRINCIPAL] I5 + P5 KAFKA STREAMING + P2 TENSORRT"
ORIGEM="${ORIGEM:-mic_balcao_1}"
ORIGEM_PASSED=false
for arg in "$@"; do
    if [ "$arg" = "--origem" ]; then
        ORIGEM_PASSED=true
        break
    fi
done

EXTRA_ARGS=()
if [ "$ORIGEM_PASSED" = false ]; then
    EXTRA_ARGS=(--origem "$ORIGEM")
fi

echo "=================================================================="
echo "  Entrada de Áudio : Kafka B2 -> Tópico 'ods.p5.audio_chunk' (P5)"
echo "  Origem no Kafka  : $ORIGEM"
echo "  VAD & Wake Word  : Silero VAD + KWS ('Lexie') na CPU"
echo "  Inferência ASR   : P2 Runtime (TensorRT Whisper) na GPU"
echo "  Saída / Eventos  : Kafka -> Tópico 'ods.i5.transcript' (A7)"
echo "  Python Utilizado : $PYTHON_EXEC"
echo "=================================================================="

exec "$PYTHON_EXEC" src/pipeline.py \
    --kafka \
    --publish-kafka \
    --wake-word \
    --wake-words lexie \
    "${EXTRA_ARGS[@]}" \
    "$@"
