#!/usr/bin/env bash
# ==============================================================================
# MENU CENTRAL DE EXECUÇÃO: COMPONENTE I5 (ÁUDIO, VAD & TRANSCRIÇÃO)
# ==============================================================================

set -e

I5_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$I5_DIR/scripts"

show_menu() {
    clear 2>/dev/null || true
    echo "=================================================================="
    echo "   COMPONENTE I5 — INFERÊNCIA DE VOZ, VAD & TRANSCRIÇÃO"
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
    echo "  [0] Sair"
    echo "=================================================================="
}

OPCAO="$1"

if [ -z "$OPCAO" ]; then
    show_menu
    read -r -p "Selecione uma opção [0-4]: " OPCAO
fi

case "$OPCAO" in
    1)
        shift 2>/dev/null || true
        exec "$SCRIPTS_DIR/1_principal_p5_kafka_p2.sh" "$@"
        ;;
    2)
        shift 2>/dev/null || true
        exec "$SCRIPTS_DIR/2_teste_microfone_local.sh" "$@"
        ;;
    3)
        shift 2>/dev/null || true
        exec "$SCRIPTS_DIR/3_teste_audio_pronto_whisper.sh" "$@"
        ;;
    4)
        shift 2>/dev/null || true
        exec "$SCRIPTS_DIR/4_teste_enviar_json_kafka.sh" "$@"
        ;;
    0|q|Q|sair|exit)
        echo "Encerrando."
        exit 0
        ;;
    *)
        echo "Opção inválida: '$OPCAO'"
        exit 1
        ;;
esac
