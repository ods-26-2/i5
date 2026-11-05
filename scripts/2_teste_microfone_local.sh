#!/usr/bin/env bash
# ==============================================================================
# CENÁRIO 2.1 (TESTES): Escuta Contínua do Microfone do PC Atual
# ==============================================================================
# Substitui apenas a fonte P5 pelo microfone do computador atual, mantendo todo
# o restante 100% oficial do I5:
# 1. Captura contínua do microfone local (16kHz mono via arecord/sox/parec).
# 2. Pré-processamento e Silero VAD na CPU.
# 3. KeywordSpotter da Ester com detecção de palavra de ativação ("Lexie").
# 4. Transcrição Whisper na GPU via P2 Runtime.
# 5. Publicação no Kafka (ods.i5.transcript) e salvamento em i5/jsons_gerados/.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
I5_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$I5_DIR"

if [ -f "$I5_DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$I5_DIR/.venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

export PYTHONPATH="$I5_DIR"

echo "=================================================================="
echo "   [CENÁRIO 2.1 - TESTE] MICROFONE LOCAL DO PC -> PIPELINE I5"
echo "=================================================================="
echo "  Entrada de Áudio : Microfone Local (16kHz Mono)"
echo "  Pipeline Oficial : Preprocessador + VAD + Wake Word + Whisper P2"
echo "  Saída / Eventos  : Publicação Kafka (ods.i5.transcript) e JSON"
echo "  Python Utilizado : $PYTHON_EXEC"
echo "------------------------------------------------------------------"
echo "  Dica: Fale 'Lexie, quero um X-Salada' para disparar o pedido."
echo "  Pressione Ctrl+C a qualquer momento para encerrar."
echo "=================================================================="

exec "$PYTHON_EXEC" teste_microfone_local/testar_vad_microfone.py \
    --wake-word \
    --wake-words lexie \
    --publish-kafka \
    "$@"
