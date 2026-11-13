# ODS I5 — Inferência de Voz, VAD & Transcrição ASR
> **Oficina de Desenvolvimento de Software (ODS 2026/2) — UFSJ**  
> **Squad:** VER-4 (Varejo)  
> **Liderança:** Mateus de Oliveira Brito  
> **Equipe:** Mateus de Oliveira Brito, Ester Maria de Souza, Rafael Andrade Blom Gurgel  
> **Componente:** `ods-i5-audio-transcription` (Camada 2 — Inferência)

---

## 1. Visão Geral

O componente **I5** é o motor de inferência de áudio da solução de varejo. Ele é responsável por:
1. **Ingestão contínua de áudio** (via barramento Kafka `ods.p5.audio_chunk` vindo da Camada 1 P5, ou do microfone local).
2. **Filtragem acústica e VAD (Voice Activity Detection):** Segmentação de fala na CPU via Silero VAD (ONNX Runtime) com histerese e filtros passa-banda Butterworth.
3. **Keyword Spotting (Wake Word):** Detecção de palavra-chave de ativação (*"Lexie"*) na CPU, descartando ruídos e conversas de fundo sem acionar a GPU.
4. **Transcrição ASR (Whisper na GPU / TensorRT):** Transcrição de fala para texto em português com cálculo de confiança e sinalização de ambiguidade (*Dúvida Explícita - RF07*).
5. **Publicação Canônica (Kafka B1/B3):** Emissão de eventos estruturados em `ods.i5.transcript` para consumo pela aplicação de varejo A7.

---

## 2. Restrições Arquiteturais na Jetson (Orin Nano / Nano)

* **GPU vs. CPU (RNF01):** Na Jetson (8 GB RAM compartilhada), a GPU é dedicada prioritariamente à inferência pesada (P2 / TensorRT Whisper). Todo o pré-processamento, Silero VAD e Keyword Spotter rodam estritamente na **CPU**.
* **Sem Mídia Bruta no Barramento de Aplicação (RNF03):** O I5 nunca envia áudio bruto para a Camada 4 (A7). A saída é estritamente um payload JSON padronizado com texto transcrito, pontuação de confiança e identificador de mesa/microfone (`origem`).
* **Latência Alvo e Streaming:** Lotes de áudio são processados com chunks de 100 ms para manter tempo de resposta quase instantâneo.

---

## 3. Guia Rápido com Docker (Recomendado)

O I5 é 100% conteinerizado com suporte nativo a GPU NVIDIA, áudio ALSA e comunicação Kafka.

### 3.1. Pré-requisitos
* **Docker** e **Docker Compose** instalados.
* **NVIDIA Container Toolkit** instalado no host (para acesso à GPU/CUDA):
  ```bash
  # Na Jetson, já vem pré-instalado com o JetPack.
  # Em Linux x86_64:
  sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```
* Adicione seu usuário ao grupo de áudio e docker:
  ```bash
  sudo usermod -aG audio,docker $USER
  ```

---

### 3.2. Execução no PC de Desenvolvimento (x86_64)

1. **Construir a imagem:**
   ```bash
   docker compose build ods-i5
   ```

2. **Abrir o Menu Interativo (`run.sh`) dentro do container:**
   ```bash
   docker compose run --rm -it ods-i5
   ```

3. **Executar um cenário específico diretamente:**
   ```bash
   # Cenário 1 (Produção: Consome P5 do Kafka e transcreve na GPU):
   docker compose run --rm ods-i5 1

   # Cenário 2 (Microfone do computador + Wake Word 'Lexie' + GPU):
   docker compose run --rm -it ods-i5 2 --origem mic_balcao_1

   # Cenário 3 (Processar arquivo de áudio de teste de data/):
   docker compose run --rm -it ods-i5 3

   # Cenário 4 (Replay de JSON direto para o Kafka):
   docker compose run --rm -it ods-i5 4 --origem mesa_1
   ```

---

### 3.3. Execução na NVIDIA Jetson (ARM64 / JetPack)

Na Jetson, utilize a imagem otimizada para arquitetura Tegra (`Dockerfile.jetson`):

1. **Construir a imagem para Jetson:**
   ```bash
   docker compose --profile jetson build ods-i5-jetson
   ```

2. **Iniciar o container com runtime NVIDIA:**
   ```bash
   docker compose --profile jetson run --rm -it ods-i5-jetson
   ```

---

## 4. Instalação e Execução Nativa (Sem Docker)

Caso prefira rodar diretamente no sistema operacional da Jetson ou PC:

### 4.1. Instalar Dependências de Sistema
```bash
sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    librdkafka-dev \
    alsa-utils \
    pulseaudio-utils \
    ffmpeg \
    git \
    build-essential
```

### 4.2. Criar Ambiente Virtual e Instalar Pacotes
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3. Executar pelo Menu Central
O I5 possui um menu de execução automatizado:
```bash
chmod +x run.sh scripts/*.sh
./run.sh
```

---

## 5. Cenários de Execução

O I5 disponibiliza 4 cenários modulares:

| Opção | Cenário | Descrição |
| :---: | :--- | :--- |
| **`1`** | **Principal (P5 Kafka + P2 GPU)** | Escuta ativa do tópico `ods.p5.audio_chunk`, VAD e KWS na CPU, Whisper na GPU, publicação em `ods.i5.transcript`. |
| **`2`** | **Microfone Local do PC** | Captura do microfone físico do computador via ALSA/PulseAudio, VAD e Keyword Spotter *"Lexie"*, transcrição GPU e Kafka. |
| **`3`** | **Áudio Pronto para Whisper** | Lê arquivos `.ogg` ou `.wav` da pasta `data/`, processa pelo pipeline completo e envia ao Kafka. |
| **`4`** | **Replay JSON Direto para o Kafka** | Reenvia um evento já gerado em `jsons_gerados/` ou permite digitar uma frase manual sem rodar inferência de áudio. |

### Dica: Alternar o Microfone/Mesa de Destino
Para mudar a mesa de destino do pedido (`origem`) no Kafka:
* No menu interativo, digite **`M`** para escolher (`mic_balcao_1`, `mic_balcao_2`, `mesa_1`, `drive_thru`, etc.).
* Ou passe diretamente via flag: `./run.sh 2 --origem mic_balcao_2`.

---

## 6. Configurações (`config/pipeline_config.yaml`)

Principais opções configuráveis:
* `kafka.bootstrap_servers`: Endereço do broker Kafka (ex: `34.66.235.91:9093` ou `172.18.X.X:9092`).
* `vad.speech_threshold`: Limiar de detecção de voz do Silero VAD (padrão: `0.50`).
* `whisper.model_name`: Modelo Whisper utilizado (ex: `openai/whisper-small`).
* `whisper.device`: Dispositivo alvo (`cuda` para GPU Jetson / `cpu` para fallback).

---

## 7. Estrutura de Diretórios

```text
i5/
├── config/
│   └── pipeline_config.yaml      # Parâmetros de VAD, Wake Word, Whisper e Kafka
├── data/                         # Áudios de teste (.ogg, .wav)
├── jsons_gerados/                # Histórico de eventos transcritos salvos em JSON
├── scripts/
│   ├── 1_principal_p5_kafka_p2.sh
│   ├── 2_teste_microfone_local.sh
│   ├── 3_teste_audio_pronto_whisper.sh
│   └── 4_teste_enviar_json_kafka.sh
├── src/
│   ├── audio_preprocessor.py     # Filtros passa-banda e normalização
│   ├── vad_segmenter.py          # Silero VAD (ONNX Runtime - CPU)
│   ├── keyword_spotter.py        # Detecção de Wake Word "Lexie" (CPU)
│   ├── p2_runtime.py             # Runtime acelerado P2 TensorRT / Whisper (GPU)
│   ├── TranscriptionWhisper.py   # Motor Whisper ASR
│   ├── kafka_publisher.py        # Publicador de eventos canônicos no Kafka
│   └── pipeline.py               # Orquestrador do pipeline de streaming
├── teste_microfone_local/        # Módulo de teste contínuo com microfone físico
├── tests/                        # Testes unitários e de integração
├── Dockerfile                    # Container Docker padrão (x86_64 / PC)
├── Dockerfile.jetson             # Container Docker otimizado para NVIDIA Jetson
├── docker-compose.yml            # Orquestração de containers e mapeamento de GPU/áudio
├── requirements.txt              # Dependências Python homologadas
└── run.sh                        # Menu central de execução dos cenários
```
