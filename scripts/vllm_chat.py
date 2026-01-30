from __future__ import annotations

import argparse
import os
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive CLI chat with a local vLLM OpenAI-compatible server.")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8002/v1"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Reply with final answer only; no <think> tags.",
        }
    ]

    print(f"vLLM URL: {url}")
    print(f"Model: {args.model}")
    print("Commands: /reset (clear history), /exit (quit)\n")

    with httpx.Client(timeout=600) as client:
        while True:
            try:
                user = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return 0

            if not user:
                continue
            if user in {"/exit", "/quit"}:
                return 0
            if user == "/reset":
                messages = messages[:1]
                print("History cleared.\n")
                continue

            messages.append({"role": "user", "content": user})
            resp = client.post(
                url,
                headers=headers,
                json={
                    "model": args.model,
                    "messages": messages,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                },
            )

            if resp.status_code == 400:
                print(f"LLM> (error 400) {resp.text.strip()}\n")
                print("Tip: try /reset or restart vLLM with a larger --max-model-len.\n")
                continue

            resp.raise_for_status()
            assistant = resp.json()["choices"][0]["message"]["content"]
            print(f"LLM> {assistant}\n")
            messages.append({"role": "assistant", "content": assistant})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
