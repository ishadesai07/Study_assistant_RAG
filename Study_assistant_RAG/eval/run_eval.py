"""Compare chunk size / overlap / top-k / embedding model on a labelled question set.

Retrieval metrics need no API key:
  page_hit@k   - an expected page appears among the top-k chunks
  evidence@k   - some top-k chunk contains ALL evidence strings (answer text was retrieved)
  MRR          - 1/rank of the first chunk containing the evidence
Optional --judge adds end-to-end answer correctness graded by an LLM (needs an API key).

Usage: python -m eval.run_eval [--pdf PATH] [--questions PATH] [--judge] [--out eval/results.md]
"""
import argparse
import itertools
import json
import re
import time
import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from app import ingest
from app.config import ROOT, settings

KS = [1, 2, 4, 8]
CHUNK_CONFIGS = [(128, 16), (256, 32), (512, 64), (1024, 128)]  # (size, overlap)
EMBED_MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]
DEFAULT_EMBED = EMBED_MODELS[0]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("-\n", "").replace("\n", " ")).lower()


def retrieve_ranked(docs, size, overlap, embed, questions):
    nodes = ingest.split(docs, size, overlap)
    col = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False)).create_collection(f"e{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    index = VectorStoreIndex(
        nodes, storage_context=StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=col)), embed_model=embed
    )
    retriever = index.as_retriever(similarity_top_k=max(KS))
    return len(nodes), [retriever.retrieve(q["question"]) for q in questions]


def score(questions, ranked):
    m = {f"page@{k}": 0.0 for k in KS} | {f"evid@{k}": 0.0 for k in KS} | {"mrr": 0.0}
    for q, hits in zip(questions, ranked):
        ev = [norm(e) for e in q["evidence"]]
        first = None
        for r, h in enumerate(hits, 1):
            if all(e in norm(h.get_content()) for e in ev):
                first = first or r
        for k in KS:
            top = hits[:k]
            m[f"page@{k}"] += any(int(h.metadata["page"]) in q["pages"] for h in top)
            m[f"evid@{k}"] += first is not None and first <= k
        m["mrr"] += 1 / first if first else 0
    return {k: v / len(questions) for k, v in m.items()}


JUDGE_SYS = "You grade answers. Reply with exactly one word: CORRECT if the candidate answer agrees with the reference on all key facts, otherwise INCORRECT."


def judge_run(questions, ranked, top_k):
    from app.llm import get_llm
    from app.rag import ANSWER_SYSTEM, Hit, _context

    llm = get_llm()
    if llm is None:
        raise SystemExit("--judge needs ANTHROPIC_API_KEY or OPENAI_API_KEY")
    ok = 0
    for q, hits in zip(questions, ranked):
        hs = [Hit(h.get_content(), h.metadata["source"], int(h.metadata["page"]), h.score or 0) for h in hits[:top_k]]
        ans = llm.complete(ANSWER_SYSTEM, f"Context:\n{_context(hs)}\n\nQuestion: {q['question']}")
        verdict = llm.complete(JUDGE_SYS, f"Question: {q['question']}\nReference: {q['expected_answer']}\nCandidate: {ans}", 5)
        ok += verdict.strip().upper().startswith("CORRECT")
    return ok / len(questions)


def table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    return head + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "data/sample/attention_is_all_you_need.pdf"))
    ap.add_argument("--questions", default=str(ROOT / "eval/questions.json"))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "eval/results.md"))
    a = ap.parse_args()

    questions = json.load(open(a.questions))
    docs = ingest.load_pdf(a.pdf)
    embeds = {name: HuggingFaceEmbedding(model_name=name) for name in EMBED_MODELS}
    results = []

    def run(label, size, overlap, name):
        t = time.time()
        n, ranked = retrieve_ranked(docs, size, overlap, embeds[name], questions)
        s = score(questions, ranked)
        row = {"config": label, "chunks": n, **{k: f"{v:.2f}" for k, v in s.items()}}
        if a.judge:
            row["answer_acc@4"] = f"{judge_run(questions, ranked, 4):.2f}"
        print(row, f"{time.time() - t:.0f}s", flush=True)
        results.append(row)

    for size, ov in CHUNK_CONFIGS:
        run(f"chunk={size}/{ov}, bge-small", size, ov, DEFAULT_EMBED)
    best = max(results, key=lambda r: (float(r["evid@4"]), float(r["mrr"])))
    bs, bo = re.search(r"chunk=(\d+)/(\d+)", best["config"]).groups()
    run(f"chunk={bs}/{bo}, MiniLM-L6", int(bs), int(bo), EMBED_MODELS[1])

    cols = ["config", "chunks"] + [f"page@{k}" for k in KS] + [f"evid@{k}" for k in KS] + ["mrr"] + (["answer_acc@4"] if a.judge else [])
    md = f"# Evaluation results\n\n{len(questions)} questions over `{a.pdf.split('/')[-1]}` ({len(docs)} pages).\n\n" + table(results, cols)
    md += f"\nBest chunking by evid@4 then MRR: **{best['config']}**.\n"
    open(a.out, "w").write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
