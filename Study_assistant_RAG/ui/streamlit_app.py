import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Study Assistant", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.messages = []


def call(method, path, **kw):
    r = requests.request(method, f"{API}{path}", timeout=300, **kw)
    if not r.ok:
        st.error(r.json().get("detail", r.text))
        return None
    return r.json()


with st.sidebar:
    st.header("Course material")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if pdf and st.button("Ingest PDF"):
        with st.spinner("Chunking and embedding…"):
            res = call("POST", "/ingest/pdf", files={"file": (pdf.name, pdf.getvalue(), "application/pdf")})
        if res:
            st.success(f"{res['source']}: {res['pages']} pages → {res['chunks']} chunks")
    url = st.text_input("…or a URL")
    if url and st.button("Ingest URL"):
        with st.spinner("Fetching…"):
            res = call("POST", "/ingest/url", json={"url": url})
        if res:
            st.success(f"{res['chunks']} chunks")
    srcs = call("GET", "/sources") or []
    for s in srcs:
        st.caption(f"📄 {s['source']} ({s['chunks']} chunks)")
    scope = st.selectbox("Limit to source", ["All"] + [s["source"] for s in srcs])
    if st.button("New conversation"):
        st.session_state.session_id, st.session_state.messages = None, []
        st.rerun()

chat_tab, quiz_tab = st.tabs(["💬 Ask", "📝 Quiz me"])

with chat_tab:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for c in m.get("citations", []):
                if c["cited"]:
                    with st.expander(f"[{c['id']}] {c['source']} — p.{c['page']}"):
                        st.write(c["snippet"] + "…")
    if q := st.chat_input("Ask about your material"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Thinking…"):
            res = call("POST", "/ask", json={"question": q, "session_id": st.session_state.session_id, "source": None if scope == "All" else scope})
        if res:
            st.session_state.session_id = res["session_id"]
            st.session_state.messages.append({"role": "assistant", "content": res["answer"], "citations": res["citations"]})
        st.rerun()

with quiz_tab:
    topic = st.text_input("Topic (optional — blank samples the whole material)")
    n = st.slider("Questions", 1, 10, 5)
    if st.button("Generate quiz"):
        with st.spinner("Writing questions…"):
            res = call("POST", "/quiz", json={"n": n, "topic": topic or None, "source": None if scope == "All" else scope})
        if res:
            st.markdown(res["quiz"])
