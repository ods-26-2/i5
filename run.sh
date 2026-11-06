#!/usr/bin/env bash
# ==============================================================================
# MENU CENTRAL DE EXECUÇÃO: COMPONENTE I5 (ÁUDIO, VAD & TRANSCRIÇÃO)
# ==============================================================================

set -e

I5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$I5_DIR/scripts"

ORIGEM="${ORIGEM:-mic_balcao_1}"
POSITIONAL_ARGS=()

# Processa parâmetros de linha de comando (--origem ou -o)
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
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

trocar_origem() {
    echo ""
    echo "=================================================================="
    echo "  SELEÇÃO DO MICROFONE / MESA DE DESTINO DO PEDIDO"
    echo "=================================================================="
    echo "  [1] mic_balcao_1  (Balcão 1 - Padrão)"
    echo "  [2] mic_balcao_2  (Balcão 2)"
    echo "  [3] mesa_1        (Mesa 1)"
    echo "  [4] mesa_2        (Mesa 2)"
    echo "  [5] drive_thru    (Drive-thru)"
    echo "  [6] Digitar outro nome personalizado"
    echo "------------------------------------------------------------------"
    read -r -p "Escolha o microfone/mesa [1-6 ou Enter para manter '$ORIGEM']: " SEL_ORIGEM
    case "$SEL_ORIGEM" in
        1) ORIGEM="mic_balcao_1" ;;
        2) ORIGEM="mic_balcao_2" ;;
        3) ORIGEM="mesa_1" ;;
        4) ORIGEM="mesa_2" ;;
        5) ORIGEM="drive_thru" ;;
        6)
            read -r -p "Digite o identificador (ex: totem_1, caixa_1): " CUSTOM_ORIGEM
            ORIGEM="${CUSTOM_ORIGEM:-$ORIGEM}"
            ;;
        *) ;;
    esac
    echo ">> Microfone/Mesa selecionado: $ORIGEM"
    echo ""
}

show_menu() {
    clear 2>/dev/null || true
    echo "=================================================================="
    echo "   COMPONENTE I5 — INFERÊNCIA DE VOZ, VAD & TRANSCRIÇÃO"
    echo "   Microfone / Mesa Ativa: [ $ORIGEM ]"
    echo "=================================================================="
    echo ""
    echo "  [1] CENÁRIO PRINCIPAL (Produção / Jetson)"
    echo "      -> Escuta ativa de P5 via Kafka + VAD + KWS ('Lexie') + P2 (Whisper)"
    echo ""
    echo "  [2] TESTE: Escuta do Microfone do PC Atual"
    echo "      -> Captura do microfone local + VAD + Wake Word 'Lexie' + P2 + Kafka"
    echo ""
    echo "  [3] TESTE: Enviar Áudio Pronto para Whisper"
    echo "      -> Processa arquivo .ogg/.wav pelo pipeline I5 e publica no Kafka"
    echo ""
    echo "  [4] TESTE: Enviar JSON Pronto Direto para o Kafka"
    echo "      -> Reenvia JSON de jsons_gerados/ ou digita frase manual para Kafka"
    echo ""
    echo "  [M] Alterar Microfone / Mesa de Destino (atual: $ORIGEM)"
    echo "  [0] Sair"
    echo "=================================================================="
}

OPCAO="$1"

while true; do
    if [ -z "$OPCAO" ]; then
        show_menu
        read -r -p "Selecione uma opção [0-4, M]: " OPCAO
    fi

    case "$OPCAO" in
        m|M|microfone|origem)
            trocar_origem
            OPCAO=""
            ;;
        1)
            shift 2>/dev/null || true
            export ORIGEM
            exec "$SCRIPTS_DIR/1_principal_p5_kafka_p2.sh" --origem "$ORIGEM" "$@"
            ;;
        2)
            shift 2>/dev/null || true
            export ORIGEM
            exec "$SCRIPTS_DIR/2_teste_microfone_local.sh" --origem "$ORIGEM" "$@"
            ;;
        3)
            shift 2>/dev/null || true
            export ORIGEM
            exec "$SCRIPTS_DIR/3_teste_audio_pronto_whisper.sh" --origem "$ORIGEM" "$@"
            ;;
        4)
            shift 2>/dev/null || true
            export ORIGEM
            exec "$SCRIPTS_DIR/4_teste_enviar_json_kafka.sh" --origem "$ORIGEM" "$@"
            ;;
        0|q|Q|sair|exit)
            echo "Encerrando."
            exit 0
            ;;
        *)
            echo "Opção inválida: '$OPCAO'"
            OPCAO=""
            ;;
    esac
done
