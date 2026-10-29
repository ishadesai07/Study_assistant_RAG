"""Smoke test: ingest the sample PDF into a temp store and ask a question (no API key needed)."""
import os
import tempfile

os.environ["CHROMA_DIR"] = tempfile.mkdtemp()
os.environ["UPLOAD_DIR"] = tempfile.mkdtemp()
os.environ["LLM_PROVIDER"] = "none"

from fastapi.testclient import TestClient  # noqa: E402

from app.config import ROOT  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
PDF = ROOT / "data/sample/attention_is_all_you_need.pdf"


def test_end_to_end():
    with PDF.open("rb") as f:
        r = client.post("/ingest/pdf", files={"file": (PDF.name, f, "application/pdf")})
    assert r.status_code == 200 and r.json()["chunks"] > 0
    assert client.get("/sources").json()[0]["source"] == PDF.name

    r = client.post("/ask", json={"question": "How many layers are in the encoder stack?"})
    body = r.json()
    assert body["citations"] and 3 in {c["page"] for c in body["citations"]}

    r2 = client.post("/ask", json={"question": "And the decoder?", "session_id": body["session_id"]})
    assert r2.status_code == 200
