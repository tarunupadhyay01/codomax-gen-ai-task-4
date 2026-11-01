"""Chat With Your Documents - Streamlit RAG app.

Run with:  streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

from rag_core import (
    MAX_CHUNKS,
    answer,
    build_chunks,
    get_client,
    index_documents,
    load_document,
    retrieve,
)

load_dotenv()
st.set_page_config(page_title="Chat With Your Documents", page_icon="📄")
st.title("📄 Chat With Your Documents")
st.caption("RAG app | Gemini embeddings + FAISS vector database | Codomax Digital Solutions - Module 4")


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:  # noqa: BLE001
        return None


try:
    client = get_client(get_api_key())
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# ---------- Session state ----------
st.session_state.setdefault("store", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("stats", None)

# ---------- Sidebar: upload and settings ----------
with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader("PDF, TXT or MD files", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, 50)
    overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)

    if st.button("Process documents", type="primary", disabled=not files):
        try:
            with st.spinner("Reading, chunking and embedding..."):
                pages = []
                for f in files:
                    pages.extend(load_document(f.name, f.getvalue()))
                chunks = build_chunks(pages, chunk_size, overlap)
                if not chunks:
                    raise ValueError("No readable text found. Scanned PDFs need OCR.")
                if len(chunks) > MAX_CHUNKS:
                    st.warning(f"Large input: only the first {MAX_CHUNKS} chunks are used.")
                    chunks = chunks[:MAX_CHUNKS]
                st.session_state.store = index_documents(client, chunks)
                st.session_state.stats = {"files": len(files), "pages": len(pages), "chunks": len(chunks)}
                st.session_state.messages = []
            st.success("Documents are ready. Ask your questions!")
        except (RuntimeError, ValueError) as e:
            st.error(str(e))

    if st.session_state.stats:
        s = st.session_state.stats
        st.write(f"Files: **{s['files']}** | Pages: **{s['pages']}** | Chunks: **{s['chunks']}**")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat area ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for chunk, score in m["sources"]:
                    st.markdown(f"**{chunk['source']}, page {chunk['page']}** (similarity {score:.2f})")
                    st.caption(chunk["text"])

if st.session_state.store is None:
    st.info("Upload one or more documents in the sidebar and click **Process documents** to begin.")
elif question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                hits = retrieve(client, st.session_state.store, question, top_k)
                reply = answer(client, question, hits)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply, "sources": hits})
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
                st.session_state.messages.pop()
