# 🎙️ Entrada de Microfone Local para o Pipeline Oficial I5 (Substituindo apenas o P5)

O script [testar_vad_microfone.py](file:///home/mateus/Documentos/UFSJ/ODS/i5/teste_microfone_local/testar_vad_microfone.py) substitui **exclusivamente** a leitura de áudio do P5 (`simulate_audio_stream` / `KafkaAudioConsumer`) pela captura contínua do seu microfone local (`16.000 Hz` mono) e passa o gerador de chunks diretamente para `pipeline.process_stream(stream)` do [`RealtimeSpeechPipeline`](file:///home/mateus/Documentos/UFSJ/ODS/i5/src/pipeline.py#L90-L473).

Toda a cadeia oficial já implementada no repositório **I5** é executada sem nenhuma modificação:
1. [`AudioPreprocessor`](file:///home/mateus/Documentos/UFSJ/ODS/i5/src/audio_preprocessor.py) (filtro passa-banda em CPU)
2. [`VADSegmenter`](file:///home/mateus/Documentos/UFSJ/ODS/i5/src/vad_segmenter.py) (Silero VAD em CPU)
3. [`KeywordSpotter`](file:///home/mateus/Documentos/UFSJ/ODS/i5/src/keyword_spotter.py) (lógica oficial de Wake Word `"lexie"` da Ester)
4. [`TranscritorWhisper`](file:///home/mateus/Documentos/UFSJ/ODS/i5/src/TranscriptionWhisper.py) (Whisper ASR)
5. [`TranscriptKafkaPublisher`](file:///home/mateus/Documentos/UFSJ/ODS/i5/src/kafka_publisher.py) (publicação no Kafka `ods.i5.transcript` + gravação em `jsons_gerados/`)

---

## Como Rodar (Microfone -> I5 Oficial -> Kafka `ods.i5.transcript`)

```bash
cd /home/mateus/Documentos/UFSJ/ODS/i5
.venv/bin/python teste_microfone_local/testar_vad_microfone.py
```
*(Por padrão já inicia com `--wake-word` e `--publish-kafka` ativos, exatamente como na produção).*

### Flags Opcionais (mesmas do `src/pipeline.py`)
* `--origem mic_balcao_1` — Define a origem no payload enviado ao Kafka.
* `--no-publish-kafka` — Caso queira testar sem publicar no broker Kafka.
* `--no-wake-word` — Caso queira desativar o `KeywordSpotter` da Ester.
* `--threshold 0.5` / `--silence-threshold-ms 700` — Sobrescreve os limiares do `VADSegmenter`.
