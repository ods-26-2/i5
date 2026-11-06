#!/usr/bin/env bash
# ==============================================================================
# CENÁRIO 2.2 (TESTES): Enviar Áudio Pronto (.ogg / .wav) para o Whisper (P2)
# ==============================================================================
# Processa um arquivo de áudio pré-gravado através do pipeline I5:
# 1. Pré-processamento e VAD.
# 2. Transcrição com Whisper via P2 Runtime.
# 3. Geração do payload canônico e publicação no Kafka (ods.i5.transcript).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
I5_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ORIGEM="${ORIGEM:-mic_balcao_1}"
AUDIO_ARG=""
REMAINING_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --origem|-o)
            ORIGEM="$2"
            shift 2
            ;;
        --origem=*)
            ORIGEM="${1#*=}"
            shift
            ;;
        *)
            if [ -z "$AUDIO_ARG" ] && [[ "$1" != -* ]]; then
                AUDIO_ARG="$1"
            else
                REMAINING_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

AUDIO_FILE=""

if [ -n "$AUDIO_ARG" ]; then
    if [ -f "$AUDIO_ARG" ]; then
        AUDIO_FILE="$(realpath "$AUDIO_ARG")"
    elif [ -f "$I5_DIR/$AUDIO_ARG" ]; then
        AUDIO_FILE="$(realpath "$I5_DIR/$AUDIO_ARG")"
    elif [ -f "$I5_DIR/data/$AUDIO_ARG" ]; then
        AUDIO_FILE="$(realpath "$I5_DIR/data/$AUDIO_ARG")"
    fi
fi

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

if [ -z "$AUDIO_FILE" ]; then
    echo "=================================================================="
    echo "   [CENÁRIO 2.2 - TESTE] ENVIAR ÁUDIO PRONTO PARA WHISPER (P2)"
    echo "=================================================================="
    echo "Áudios de teste disponíveis em 'i5/data/':"
    
    shopt -s nullglob
    AUDIOS=("$I5_DIR/data/"*.ogg "$I5_DIR/data/"*.wav "$I5_DIR/data/"*.mp3)
    
    if [ ${#AUDIOS[@]} -eq 0 ]; then
        echo "Nenhum arquivo encontrado em i5/data/."
        read -r -p "Digite o caminho completo ou relativo do áudio: " AUDIO_FILE
    else
        for i in "${!AUDIOS[@]}"; do
            ARQ_NOME="$(basename "${AUDIOS[$i]}")"
            echo "  [$((i + 1))] $ARQ_NOME"
        done
        echo ""
        read -r -p "Escolha o número do áudio [1-${#AUDIOS[@]}] ou informe o caminho: " ESCOLHA

        if [ "$ESCOLHA" = "q" ] || [ "$ESCOLHA" = "Q" ] || [ "$ESCOLHA" = "sair" ]; then
            echo "Cancelado."
            exit 0
        elif [[ "$ESCOLHA" =~ ^[0-9]+$ ]] && [ "$ESCOLHA" -ge 1 ] && [ "$ESCOLHA" -le "${#AUDIOS[@]}" ]; then
            AUDIO_FILE="${AUDIOS[$((ESCOLHA - 1))]}"
        else
            AUDIO_FILE="$ESCOLHA"
        fi
    fi
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Erro: Arquivo '$AUDIO_FILE' não encontrado."
    exit 1
fi

echo "------------------------------------------------------------------"
echo "Processando áudio: $AUDIO_FILE"
echo "Origem no Kafka  : $ORIGEM"
echo "Publicando resultado em ods.i5.transcript..."
echo "------------------------------------------------------------------"

exec "$PYTHON_EXEC" src/pipeline.py "$AUDIO_FILE" \
    --publish-kafka \
    --origem "$ORIGEM" \
    "${REMAINING_ARGS[@]}"
