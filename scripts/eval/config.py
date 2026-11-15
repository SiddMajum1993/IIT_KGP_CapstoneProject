"""Endpoints, model names and credentials, read from the environment or a .env file in the repo root.

Copy .env.example to .env and fill it in. .env is gitignored; never commit real keys or URLs.
Values still holding a placeholder such as <your-api-key> are treated as unset.
"""
import os

DEFAULTS = {
    "LOCAL_LLM_BASE_URL": "http://localhost:11434/v1",
    "LOCAL_LLM_MODEL": "qwen3:8b-fp16",
    "LLM_MODEL": "Qwen/Qwen3-8B",
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "gemma3:27b",
}


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def get(name: str):
    """The setting's value, its default, or None if unset or still a placeholder."""
    load_dotenv()
    v = os.environ.get(name, "").strip()
    if not v or v.startswith("<"):
        return DEFAULTS.get(name)
    return v


def secret(name: str):
    """A key from NAME, or read from the file named by NAME_FILE.

    Key files may carry a label before the key (e.g. "group 07 - <key>"), so the last field is used.
    """
    value = get(name)
    path = get(name + "_FILE")
    if not value and path:
        fields = open(os.path.expanduser(path), encoding="utf-8").read().split()
        value = fields[-1] if fields else None
    return value
