"""All settings in one place, read from the environment or the .env file in the repo root.

There are two modes, picked with COPILOT_MODE (or the --mode flag of each script):
  * local - everything runs on this machine: Ollama for the answering model, a Hugging Face model for embeddings.
            Used for development, because the course APIs are not reachable with the VPN on.
  * api   - the course APIs: Qwen3-8B for answers, bge-small-en-v1.5 for embeddings. Used for the final runs.

The judge always runs on local Ollama. Keys and certificates are only read from .env at runtime.
Values still holding a placeholder such as <your-api-key> are treated as unset.
"""
import os

DEFAULTS = {
    "COPILOT_MODE": "local",
    # local mode
    "LOCAL_LLM_BASE_URL": "http://localhost:11434/v1",
    "LOCAL_LLM_MODEL": "gemma3:27b",
    "LOCAL_EMBEDDING_MODEL": "sentence-transformers/all-mpnet-base-v2",
    # api mode
    "LLM_MODEL": "Qwen/Qwen3-8B",
    "EMBEDDING_MODEL": "bge-small-en-v1.5",
    # judge (always local)
    "JUDGE_BASE_URL": "http://localhost:11434",
    "JUDGE_MODEL": "gemma3:27b",
}

GOLD_PATH = "data/processed/gold_qa.jsonl"
CORPUS_PATH = "data/processed/master_qa.jsonl"
CHROMA_DIR = "chroma_db"
RESULTS_DIR = "results/week2"

# Generation settings shared by every run, so results are reproducible.
TEMPERATURE = 0.0
SEED = 42
MAX_TOKENS = 1024  # the course API allows at most 1024 output tokens


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


def cert(name: str):
    """Path to a CA certificate for HTTPS, or None if the endpoint does not need one."""
    path = get(name)
    return os.path.expanduser(path) if path else None
