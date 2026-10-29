"""Central settings, overridable via environment variables or a .env file."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    chroma_dir: str = os.getenv("CHROMA_DIR", str(ROOT / "storage" / "chroma"))
    upload_dir: str = os.getenv("UPLOAD_DIR", str(ROOT / "storage" / "uploads"))
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    # "anthropic", "openai", "ollama" (local, free), or "none" (extractive fallback, no API key needed)
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))


settings = Settings()
