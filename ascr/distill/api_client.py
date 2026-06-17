import os
import time


DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_MODEL = "bailian/qwen3.7-plus"


def api_settings(environ=None):
    env = environ or os.environ
    return {
        "api_key": env.get("OFOX_API_KEY", ""),
        "base_url": env.get("OFOX_BASE_URL", DEFAULT_BASE_URL),
        "model": env.get("ASCR_TEACHER_MODEL", DEFAULT_MODEL),
    }


def build_client(api_key=None, base_url=None):
    settings = api_settings()
    api_key = api_key or settings["api_key"]
    base_url = base_url or settings["base_url"]
    if not api_key:
        raise EnvironmentError("OFOX_API_KEY is not set")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("API teacher needs the openai package. Install with: python -m pip install -e '.[judge]'") from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_completion_text(client, messages, model=None, max_tokens=1024, temperature=0.0, retries=3, retry_delay=2.0):
    model = model or api_settings()["model"]
    last_error = None
    for attempt in range(int(retries)):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )
            content = response.choices[0].message.content
            if not content or not str(content).strip():
                raise ValueError("empty API response content")
            return str(content)
        except Exception as exc:
            last_error = exc
            if attempt >= int(retries) - 1:
                break
            time.sleep(float(retry_delay) * (attempt + 1))
    raise last_error

