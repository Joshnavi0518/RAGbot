import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "pdf_chunks"


def chunk_text(text, chunk_size=800, overlap=150):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap

    return chunks


def safe_id(source, page, n):
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", source)
    return f"{cleaned}-p{page}-{n}"


def pdfs_to_chunks(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for piece in chunk_text(text):
                all_chunks.append({
                    "text": piece,
                    "page": i + 1,
                    "source": uploaded_file.name,
                })
    return all_chunks


@st.cache_resource
def get_embedder():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


st.title("RAGbot")
st.write("Ask questions about uploaded PDFs. Answers come only from retrieved chunks.")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False

uploaded_files = st.file_uploader(
    "Choose PDF(s)",
    type="pdf",
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"{len(uploaded_files)} file(s) selected.")

    if st.button("Index PDFs"):
        with st.spinner("Reading PDFs and creating embeddings..."):
            all_chunks = pdfs_to_chunks(uploaded_files)

            if not all_chunks:
                st.error("No text found in the PDFs. Try a text PDF, not a scan.")
            else:
                embedder = get_embedder()
                client = get_chroma()

                try:
                    client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass

                collection = client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=embedder,
                )
                collection.add(
                    ids=[safe_id(c["source"], c["page"], n) for n, c in enumerate(all_chunks)],
                    documents=[c["text"] for c in all_chunks],
                    metadatas=[
                        {"page": c["page"], "source": c["source"]}
                        for c in all_chunks
                    ],
                )

                st.session_state.indexed = True
                st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

question = st.text_input("Ask a question about the PDFs")

if question and st.session_state.indexed:
    collection = get_chroma().get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedder(),
    )
    n_results = min(3, max(collection.count(), 1))
    if collection.count() == 0:
        st.warning("Index PDF(s) first.")
    else:
        with st.spinner("Searching and generating answer..."):
            results = collection.query(
                query_texts=[question],
                n_results=n_results,
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            st.subheader("Closest chunks")
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
                with st.expander(
                    f"Match {i} — {meta['source']} — page {meta['page']} — distance {dist:.4f}"
                ):
                    st.text(doc)

            context_blocks = [
                f"[{meta['source']}, page {meta['page']}]\n{doc}"
                for doc, meta in zip(docs, metas)
            ]
            context = "\n\n".join(context_blocks)

            prompt = f"""You are a helpful assistant.
Answer the question using ONLY the CONTEXT below.
If the answer is not in the context, say you don't know based on the uploaded documents.
Cite the source and page.

CONTEXT:
{context}

QUESTION:
{question}
"""

            st.subheader("Answer")
            if not api_key:
                st.warning("GROQ_API_KEY is missing. Add it to src/.env and rerun.")
            else:
                llm = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
                response = llm.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                st.write(response.choices[0].message.content)

elif question:
    st.warning("Index PDF(s) first.")
