"""Tiny .env loader so scripts don't depend on python-dotenv."""

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    path = path or (_REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


def make_client(base_url_env: str = "OPENAI_BASE_URL"):
    """Build an OpenAI client, honoring an optional gateway base URL from the env.

    Set OPENAI_BASE_URL in .env (e.g. a TrueFoundry / LiteLLM OpenAI-compatible
    gateway endpoint) to route every call through that gateway instead of
    api.openai.com; leave it unset for the default OpenAI API. The OpenAI SDK
    already auto-reads OPENAI_BASE_URL, so any script that calls load_env() before
    constructing a client inherits this for free — but going through this helper
    keeps it explicit and prints which endpoint is live, so a misrouted run is
    obvious in the logs.

    NOTE: gateways often require gateway-scoped model ids (e.g. "openai-main/gpt-4o"
    rather than bare "gpt-4o"). Pass those via --teacher/--judge when OPENAI_BASE_URL
    is set.
    """
    load_env()
    from openai import OpenAI

    base_url = os.environ.get(base_url_env) or None
    if base_url:
        print(f"[env] routing OpenAI calls through base_url={base_url}")
    return OpenAI(base_url=base_url) if base_url else OpenAI()
