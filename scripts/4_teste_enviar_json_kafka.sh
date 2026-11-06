#!/usr/bin/env bash
# ==============================================================================
# CENÁRIO 2.3 (TESTES): Enviar JSON Pronto Direto para o Kafka (ods.i5.transcript)
# ==============================================================================
# Publica diretamente um payload canônico de evento no tópico Kafka sem
# precisar rodar a inferência de áudio do Whisper.
# Permite:
# 1. Enviar um arquivo JSON existente da pasta i5/jsons_gerados/.
# 2. Digitar uma frase rápida que será encapsulada no envelope B1/B2 e enviada.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
I5_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ORIGEM="${ORIGEM:-mic_balcao_1}"
JSON_ARG=""
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
            if [ -z "$JSON_ARG" ] && [[ "$1" != -* ]]; then
                JSON_ARG="$1"
            else
                REMAINING_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [ -n "$JSON_ARG" ]; then
    if [ -f "$JSON_ARG" ]; then
        JSON_ARG="$(realpath "$JSON_ARG")"
    elif [ -f "$I5_DIR/$JSON_ARG" ]; then
        JSON_ARG="$(realpath "$I5_DIR/$JSON_ARG")"
    elif [ -f "$I5_DIR/jsons_gerados/$JSON_ARG" ]; then
        JSON_ARG="$(realpath "$I5_DIR/jsons_gerados/$JSON_ARG")"
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

if [ -z "$JSON_ARG" ]; then
    echo "=================================================================="
    echo "   [CENÁRIO 2.3 - TESTE] ENVIAR JSON DIRETO PARA O KAFKA"
    echo "=================================================================="
    echo "Opções disponíveis:"
    echo "  [T] Digitar uma frase de pedido manualmente para enviar agora"
    echo "------------------------------------------------------------------"
    echo "Arquivos JSON recentes em 'i5/jsons_gerados/':"

    shopt -s nullglob
    # Ordena pelos mais recentes
    mapfile -t ARQS < <(ls -t "$I5_DIR/jsons_gerados/"*.json 2>/dev/null | head -n 8)

    if [ ${#ARQS[@]} -gt 0 ]; then
        for i in "${!ARQS[@]}"; do
            NOME="$(basename "${ARQS[$i]}")"
            # Extrai preview do texto se possível
            TEXTO_PREVIEW="$(grep -o '"texto_transcrito": *"[^"]*"' "${ARQS[$i]}" 2>/dev/null | head -n 1 | cut -d'"' -f4 || echo "")"
            if [ -n "$TEXTO_PREVIEW" ]; then
                echo "  [$((i + 1))] $NOME -> \"$TEXTO_PREVIEW\""
            else
                echo "  [$((i + 1))] $NOME"
            fi
        done
    fi
    echo ""
    read -r -p "Escolha [1-${#ARQS[@]}], [T] para digitar texto, ou caminho do JSON: " ESCOLHA

    if [ "$ESCOLHA" = "q" ] || [ "$ESCOLHA" = "Q" ] || [ "$ESCOLHA" = "sair" ]; then
        echo "Cancelado."
        exit 0
    elif [ "$ESCOLHA" = "T" ] || [ "$ESCOLHA" = "t" ]; then
        read -r -p "Digite a frase do pedido (ex: Quero dois x-saladas e um suco): " FRASE
        DEFAULT_ORIGEM="${ORIGEM:-mic_balcao_1}"
        read -r -p "Mesa / Microfone de Origem [padrão: $DEFAULT_ORIGEM]: " INPUT_ORIGEM
        ORIGEM="${INPUT_ORIGEM:-$DEFAULT_ORIGEM}"

        TMP_JSON="/tmp/evento_teste_$(date +%s).json"
        cat <<EOF > "$TMP_JSON"
{
  "message_type": "event",
  "producer": "I5",
  "schema_version": "1.0",
  "payload": {
    "origem": "$ORIGEM",
    "texto_transcrito": "$FRASE",
    "confianca_geral": 0.95,
    "confidence": 0.95,
    "duvida_explicita": false
  }
}
EOF
        JSON_ARG="$TMP_JSON"
    elif [[ "$ESCOLHA" =~ ^[0-9]+$ ]] && [ "$ESCOLHA" -ge 1 ] && [ "$ESCOLHA" -le "${#ARQS[@]}" ]; then
        JSON_ARG="${ARQS[$((ESCOLHA - 1))]}"
    else
        JSON_ARG="$ESCOLHA"
    fi
fi

if [ ! -f "$JSON_ARG" ]; then
    echo "Erro: Arquivo '$JSON_ARG' não encontrado."
    exit 1
fi

echo "------------------------------------------------------------------"
echo "Publicando JSON no Kafka (ods.i5.transcript): $(basename "$JSON_ARG")"
echo "------------------------------------------------------------------"

ORIGEM_FLAG=()
if [ -n "$ORIGEM" ]; then
    ORIGEM_FLAG=(--origem "$ORIGEM")
fi

echo "Origem no Kafka  : $ORIGEM"
echo "------------------------------------------------------------------"

exec "$PYTHON_EXEC" src/pipeline.py \
    --replay-json "$JSON_ARG" \
    "${ORIGEM_FLAG[@]}" \
    "${REMAINING_ARGS[@]}"
