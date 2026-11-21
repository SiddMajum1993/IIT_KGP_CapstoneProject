"""One chat function for the answering model, in either mode (see config.py).

Both Ollama and the course API speak the OpenAI chat format, so the same client works for both;
only the URL, key and model name change.
"""
import re
import time

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

from copilot import config

RETRY_DELAYS = [2, 4, 8, 16]  # seconds, for 429 / 503 / dropped connections


def make_client(mode: str):
    """Return (client, model name, extra request fields) for the chosen mode."""
    if mode == "local":
        client = OpenAI(base_url=config.get("LOCAL_LLM_BASE_URL"), api_key="ollama")  # Ollama ignores the key
        return client, config.get("LOCAL_LLM_MODEL"), {}
    base_url, key = config.get("LLM_BASE_URL"), config.secret("LLM_API_KEY")
    assert base_url and key, "set LLM_BASE_URL and LLM_API_KEY (or LLM_API_KEY_FILE) in .env (see .env.example)"
    cert = config.cert("LLM_CA_CERT")
    client = OpenAI(base_url=base_url, api_key=key, http_client=httpx.Client(verify=cert) if cert else None)
    # Qwen3 "thinks" out loud by default; this vLLM switch turns that off so we get the answer only.
    return client, config.get("LLM_MODEL"), {"chat_template_kwargs": {"enable_thinking": False}}


def chat(client: OpenAI, model: str, extra_body: dict, messages: list) -> dict:
    """Send one chat request, retrying on rate limits and dropped connections.

    Always returns a dict: the answer plus usage numbers, or answer=None and an error message.
    """
    error = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        time.sleep(delay)
        start = time.perf_counter()
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=config.TEMPERATURE,
                                               seed=config.SEED, max_tokens=config.MAX_TOKENS, extra_body=extra_body)
        except APIStatusError as e:
            error = f"HTTP {e.status_code}"
            if e.status_code in (429, 503):
                continue
            break
        except APIConnectionError as e:
            error = f"connection: {e}"
            continue
        answer = r.choices[0].message.content or ""
        # Thinking should be off; strip any stray <think> block so it is not scored as the answer.
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        return dict(answer=answer, error=None, attempts=attempt + 1,
                    latency_s=round(time.perf_counter() - start, 2),
                    prompt_tokens=r.usage.prompt_tokens if r.usage else None,
                    completion_tokens=r.usage.completion_tokens if r.usage else None,
                    finish_reason=r.choices[0].finish_reason)
    return dict(answer=None, error=error, attempts=attempt + 1)


def stream(client: OpenAI, model: str, extra_body: dict, messages: list):
    """Same request as chat(), but yields the answer piece by piece (for the chat UI; no retries)."""
    r = client.chat.completions.create(model=model, messages=messages, temperature=config.TEMPERATURE,
                                       seed=config.SEED, max_tokens=config.MAX_TOKENS, extra_body=extra_body,
                                       stream=True)
    for chunk in r:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
