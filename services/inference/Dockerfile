# GPU inference container (scaffold). Build/run on a GPU machine.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends         python3 python3-pip git       && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir vllm==0.6.6

COPY services/inference/serve.sh /app/serve.sh
RUN chmod +x /app/serve.sh

ENV PORT=8002
EXPOSE 8002

CMD ["/app/serve.sh"]
