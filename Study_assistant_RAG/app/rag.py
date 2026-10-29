"""Retrieval, cited answering, follow-up handling and quiz generation."""
import random
import re
from dataclasses import dataclass

from .config import settings
from .ingest import get_collection, load_index
from .llm import LLMError, get_llm

ANSWER_SYSTEM = (
    "You are a study assistant. Answer ONLY from the numbered context passages. "
    "Cite every claim with its passage number like [1] or [2]. "
    "If the context does not contain the answer, say you could not find it in the material. "
    "Be concise and clear for a student."
)
CONDENSE_SYSTEM = (
    "Rewrite the user's last message as a standalone question, using the conversation for "
    "context (resolve pronouns and references). Output only the question."
)
QUIZ_SYSTEM = (
    "You write practice questions for students. Using ONLY the passages given, write the "
    "requested number of questions, each with a short model answer and the passage number "
    "it came from. Format exactly:\nQ1: ...\nA1: ... [passage number]\nQ2: ...\n..."
)


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float


def retrieve(question: str, top_k: int | None = None, collection_name: str = "study", source: str | None = None) -> list[Hit]:
    col = get_collection(collection_name)
    if col.count() == 0:
        return []
    index = load_index(col)
    filters = None
    if source:
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

        filters = MetadataFilters(filters=[ExactMatchFilter(key="source", value=source)])
    nodes = index.as_retriever(similarity_top_k=top_k or settings.top_k, filters=filters).retrieve(question)
    return [Hit(n.get_content(), n.metadata["source"], int(n.metadata["page"]), float(n.score or 0)) for n in nodes]


def _context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.source}, p.{h.page})\n{h.text}" for i, h in enumerate(hits, 1))


def _history_text(history: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history[-settings.history_turns * 2 :])


def condense(question: str, history: list[dict], llm) -> str:
    if not history or llm is None:
        return question
    try:
        out = llm.complete(CONDENSE_SYSTEM, f"Conversation:\n{_history_text(history)}\n\nLast message: {question}", 150)
    except LLMError:
        return question  # the answer step will surface the error
    return out or question


def _extractive(question: str, hits: list[Hit]) -> str:
    """No-LLM fallback: return the best-matching sentences from the top chunks."""
    q = set(re.findall(r"\w{3,}", question.lower()))
    scored = []
    for i, h in enumerate(hits, 1):
        for s in re.split(r"(?<=[.!?])\s+", h.text.replace("\n", " ")):
            overlap = len(q & set(re.findall(r"\w{3,}", s.lower())))
            if overlap:
                scored.append((overlap, i, s.strip()))
    best = sorted(scored, key=lambda t: -t[0])[:3]
    if not best:
        return "I could not find this in the uploaded material."
    return " ".join(f"{s} [{i}]" for _, i, s in best) + "\n\n(Extractive mode: no LLM available. Run Ollama or set an API key for generated answers; see README.)"


def ask(question: str, history: list[dict] | None = None, top_k: int | None = None, source: str | None = None) -> dict:
    history = history or []
    llm = get_llm()
    standalone = condense(question, history, llm)
    hits = retrieve(standalone, top_k, source=source)
    if not hits:
        return {"answer": "No material uploaded yet — add a PDF or URL first.", "citations": [], "standalone_question": standalone}
    try:
        answer = llm.complete(ANSWER_SYSTEM, f"Context:\n{_context(hits)}\n\nQuestion: {standalone}") if llm else _extractive(standalone, hits)
    except LLMError as e:
        answer = f"{_extractive(standalone, hits)}\n\n⚠️ {e}. Showing extractive answer instead."
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    citations = [
        {"id": i, "source": h.source, "page": h.page, "score": round(h.score, 3), "snippet": h.text[:300], "cited": i in cited}
        for i, h in enumerate(hits, 1)
    ]
    return {"answer": answer, "citations": citations, "standalone_question": standalone}


def quiz(n: int = 5, topic: str | None = None, source: str | None = None) -> dict:
    llm = get_llm()
    if topic:
        hits = retrieve(topic, top_k=6, source=source)
    else:
        col = get_collection("study")
        got = col.get(include=["documents", "metadatas"], where={"source": source} if source else None)
        idx = random.sample(range(len(got["documents"])), min(6, len(got["documents"]))) if got["documents"] else []
        hits = [Hit(got["documents"][i], got["metadatas"][i]["source"], int(got["metadatas"][i]["page"]), 0) for i in idx]
    if not hits:
        return {"quiz": "No material uploaded yet.", "citations": []}
    if llm is None:
        return {"quiz": "Quiz mode needs an LLM: run Ollama or set an API key (see README).", "citations": []}
    try:
        text = llm.complete(QUIZ_SYSTEM, f"Passages:\n{_context(hits)}\n\nWrite {n} questions.", 900)
    except LLMError as e:
        return {"quiz": f"⚠️ {e}", "citations": []}
    return {"quiz": text, "citations": [{"id": i, "source": h.source, "page": h.page} for i, h in enumerate(hits, 1)]}
