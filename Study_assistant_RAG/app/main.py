import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from . import ingest, rag
from .config import settings
from .llm import get_llm

app = FastAPI(title="Study Assistant", description="RAG over your course material with page-level citations.")

# in-memory conversation store: session_id -> [{role, content}]
SESSIONS: dict[str, list[dict]] = {}


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int | None = Field(None, ge=1, le=20)
    source: str | None = None


class UrlRequest(BaseModel):
    url: str


class QuizRequest(BaseModel):
    n: int = Field(5, ge=1, le=15)
    topic: str | None = None
    source: str | None = None


@app.get("/health")
def health():
    llm = get_llm()
    return {"status": "ok", "llm": llm.provider if llm else "none (extractive fallback)", "embed_model": settings.embed_model}


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    dest = Path(settings.upload_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / Path(file.filename).name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    docs = ingest.load_pdf(path)
    if not docs:
        raise HTTPException(422, "No extractable text (scanned PDF?)")
    return {"source": path.name, "pages": len(docs), "chunks": ingest.ingest_documents(docs)}


@app.post("/ingest/url")
def ingest_url(req: UrlRequest):
    try:
        docs = ingest.load_url(req.url)
    except Exception as e:
        raise HTTPException(400, f"Could not fetch URL: {e}")
    return {"source": req.url, "chunks": ingest.ingest_documents(docs)}


@app.get("/sources")
def sources():
    return ingest.list_sources()


@app.delete("/sources")
def delete_source(source: str):
    ingest.delete_source(source)
    return {"deleted": source}


@app.post("/ask")
def ask(req: AskRequest):
    sid = req.session_id or str(uuid.uuid4())
    history = SESSIONS.setdefault(sid, [])
    result = rag.ask(req.question, history, req.top_k, req.source)
    history += [{"role": "user", "content": req.question}, {"role": "assistant", "content": result["answer"]}]
    return {"session_id": sid, **result}


@app.delete("/sessions/{sid}")
def clear_session(sid: str):
    SESSIONS.pop(sid, None)
    return {"cleared": sid}


@app.post("/quiz")
def quiz(req: QuizRequest):
    return rag.quiz(req.n, req.topic, req.source)
