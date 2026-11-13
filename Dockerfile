# ==============================================================================
# Dockerfile: Componente I5 (Áudio, VAD & Transcrição Whisper)
# Arquitetura: x86_64 / Servidores Linux com GPU NVIDIA ou CPU
# ==============================================================================

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# 1. Dependências do sistema: ALSA, PulseAudio, compilação, ffmpeg e librdkafka
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    librdkafka-dev \
    alsa-utils \
    libasound2-dev \
    pulseaudio-utils \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalação das dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Cópia do código-fonte e scripts
COPY . .

# 4. Permissões de execução para scripts e menu de inicialização
RUN chmod +x run.sh scripts/*.sh

# Ponto de entrada padrão: menu interativo ou execução direta via argumentos
ENTRYPOINT ["./run.sh"]
CMD ["1"]
