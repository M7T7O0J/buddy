# Inference

This container runs vLLM in OpenAI-compatible mode.

- Endpoint: /v1/chat/completions
- Configure via env:
  - MODEL_NAME
  - PORT

For LoRA adapters, see vLLM docs on `--enable-lora` and adapter loading.
