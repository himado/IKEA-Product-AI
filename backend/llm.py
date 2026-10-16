"""
Thin wrapper around Gemini (google-genai SDK) for structured outputs.

- response_schema = a Pydantic model -> the reply is parsed and validated (response.parsed)
- retries with backoff on 5xx / 429 (gemini-3.8-flash returned "503 high demand" on the
  first test call; the retry succeeded)
- on-disk cache keyed by (model, prompt, schema) so evals can be re-run for free

Key: GOOGLE_API_KEY in .env
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"]   # tried in order
TIMEOUT_MS = 45_000   # per request; the API returned 504 after ~60 s when overloaded
CACHE_DIR = ROOT / "data" / "llm_cache"
T = TypeVar("T", bound=BaseModel)

_client = None


def client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                               http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client


class LLMUnavailable(RuntimeError):
    pass


def thinking_config(model: str, level: str) -> types.ThinkingConfig:
    # Gemini 3.x takes a thinking level; 2.5 takes a token budget (0 = off)
    if "2.5" in model:   # LOW = off (extraction); MEDIUM/HIGH = some reasoning (rerank)
        return types.ThinkingConfig(thinking_budget=0 if level in ("LOW", "MINIMAL") else 1024)
    return types.ThinkingConfig(thinking_level=level)


def structured(prompt: str, schema: Type[T], system: str, models=None,
               thinking: str = "LOW", use_cache: bool = True, retries: int = 2):
    """Structured-output call with model fallback. Returns (validated `schema` instance, model used).
    Raises LLMUnavailable if every model fails."""
    last = None
    chain = list(models or FALLBACK_MODELS)
    live = [m for m in chain if _skip_until.get(m, 0) <= time.time()] or chain[-1:]
    for model in live:
        try:
            out = _structured_one(prompt, schema, system, model, thinking, use_cache, retries)
            _skip_until.pop(model, None)
            return out, model
        except (errors.ServerError, errors.ClientError, RuntimeError) as e:
            last = e
            code = getattr(e, "code", None)
            if isinstance(e, errors.ClientError) and code not in (429, 404):
                raise
            # remember the failure: quota (429) / unknown model (404) for 10 min, overload / timeout 1 min,
            # so the next request goes straight to a model that works (saved ~15 s per request)
            _skip_until[model] = time.time() + (600 if isinstance(e, errors.ClientError) else 60)
    raise LLMUnavailable(f"all Gemini models failed: {str(last)[:200]}")


_skip_until = {}   # model -> time until which it is skipped


def _structured_one(prompt, schema, system, model, thinking, use_cache, retries):
    key = hashlib.sha256(json.dumps([model, system, prompt, schema.model_json_schema(), thinking],
                                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return schema.model_validate_json(json.loads(cache_file.read_text(encoding="utf-8"))["output"])

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema,
        temperature=0.0,
        thinking_config=thinking_config(model, thinking),
    )
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client().models.generate_content(model=model, contents=prompt, config=config)
            break
        except errors.ServerError:
            if attempt == retries - 1:
                raise                              # 5xx: overloaded -> caller tries the next model
        except errors.ClientError as e:
            # 429 RESOURCE_EXHAUSTED = daily/minute quota: waiting 3 s does not help, try the next model
            raise
        except Exception as e:                     # client-side timeout
            if attempt == retries - 1:
                raise RuntimeError(f"{model}: {type(e).__name__}") from e
        time.sleep(3 * (attempt + 1))

    parsed: Optional[T] = r.parsed if isinstance(r.parsed, schema) else None
    if parsed is None:                              # fall back to the raw JSON text
        parsed = schema.model_validate_json(r.text)
    u = r.usage_metadata
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "model": model, "prompt": prompt, "output": parsed.model_dump_json(),
        "seconds": round(time.time() - t0, 2),
        "usage": {"prompt": u.prompt_token_count, "output": u.candidates_token_count,
                  "thoughts": u.thoughts_token_count},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return parsed
