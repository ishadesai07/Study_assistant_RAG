import os
from pathlib import Path

# LlamaIndex downloads NLTK data for its splitter, and NLTK refuses group-writable cache dirs
# (which a umask of 002 creates by default). Pre-create a private one inside the project.
_root = Path(__file__).resolve().parent.parent / "storage" / "nltk"
for _d in (_root, _root / "_static", _root / "_static" / "nltk_cache"):
    _d.mkdir(parents=True, exist_ok=True)
    _d.chmod(0o700)
os.environ.setdefault("NLTK_DATA", str(_root))
# The xet download backend was extremely slow in testing; plain HTTP is much faster.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Chroma's telemetry client is broken against the installed posthog version and only logs errors.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# If the embedding model is already cached, go fully offline so a missing/slow network
# doesn't cause minutes of retries. Set HF_HUB_OFFLINE=0 to force online (e.g. to switch models).
_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# LlamaIndex's HuggingFaceEmbedding caches models in its own folder rather than the default HF cache.
_hub = Path(os.getenv("LLAMA_INDEX_CACHE_DIR") or Path.home() / ".cache" / "llama_index")
if any((_hub / f"models--{_model.replace('/', '--')}" / "snapshots").glob("*/config.json")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# chromadb 0.6 calls posthog.capture() even with telemetry disabled, and posthog>=6 changed that
# signature, producing an error log per call. Make the telemetry hook a no-op.
try:
    from chromadb.telemetry.product.posthog import Posthog as _Posthog

    _Posthog.capture = lambda self, event: None
except Exception:
    pass
