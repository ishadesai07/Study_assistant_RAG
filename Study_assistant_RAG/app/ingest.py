"""Load PDFs / URLs -> page-aware Documents -> chunk -> embed -> Chroma."""
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from pypdf import PdfReader

from .config import settings

_embed = None


def embed_model() -> HuggingFaceEmbedding:
    global _embed
    if _embed is None:
        _embed = HuggingFaceEmbedding(model_name=settings.embed_model)
    return _embed


def load_pdf(path: str | Path, source: str | None = None) -> list[Document]:
    """One Document per page so every chunk keeps an exact page number."""
    source = source or Path(path).name
    docs = []
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, metadata={"source": source, "page": i}))
    return docs


def load_url(url: str) -> list[Document]:
    r = requests.get(url, timeout=20, headers={"User-Agent": "study-assistant/0.1"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
        t.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    # web pages have no pages; use a single "page" 1 and cite the URL
    return [Document(text=text, metadata={"source": url, "page": 1, "title": title})]


def split(docs: list[Document], chunk_size: int, chunk_overlap: int):
    return SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents(docs)


def get_collection(name: str = "study", persist_dir: str | None = None):
    client = chromadb.PersistentClient(path=persist_dir or settings.chroma_dir, settings=ChromaSettings(anonymized_telemetry=False))
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def build_index(nodes, collection) -> VectorStoreIndex:
    store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex(
        nodes,
        storage_context=StorageContext.from_defaults(vector_store=store),
        embed_model=embed_model(),
    )


def load_index(collection) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=collection), embed_model=embed_model())


def ingest_documents(docs: list[Document], collection_name: str = "study") -> int:
    # re-ingesting the same source replaces its old chunks
    col = get_collection(collection_name)
    for s in {d.metadata["source"] for d in docs}:
        col.delete(where={"source": s})
    nodes = split(docs, settings.chunk_size, settings.chunk_overlap)
    build_index(nodes, col)
    return len(nodes)


def list_sources(collection_name: str = "study") -> list[dict]:
    got = get_collection(collection_name).get(include=["metadatas"])
    counts: dict[str, int] = {}
    for m in got["metadatas"]:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


def delete_source(source: str, collection_name: str = "study") -> None:
    get_collection(collection_name).delete(where={"source": source})
