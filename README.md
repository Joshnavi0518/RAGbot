# RAGbot
Live Link : https://ragbot-cdrrgnszfxuhefr7ppnnk4.streamlit.app

A simple Retrieval-Augmented Generation chatbot: upload PDFs, ask a question, and get an answer grounded in the files with source pages.

## How it works

1. Extract text from each PDF page (`pypdf`)
2. Split pages into overlapping chunks
3. Embed chunks with `all-MiniLM-L6-v2` and store them in Chroma
4. Embed the question and retrieve the closest chunks
5. Send those chunks to a Groq LLM, which must answer only from that context

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r src\requirements.txt
```

Copy `src/.env.example` to `src/.env` and add a free Groq key from [console.groq.com](https://console.groq.com):

```
GROQ_API_KEY=your_key_here
```

Do not commit `.env`.

## Run

```powershell
streamlit run src/app.py
```

1. Upload one or more text PDFs (not scanned images)
2. Click **Index PDFs**
3. Ask a question

The app shows the retrieved chunks and an answer with file/page citations. If the answer is not in the PDF, it should say it does not know.

## Stack

- UI: Streamlit
- Vector store: Chroma
- Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`)
- LLM: Groq (`openai/gpt-oss-20b`)
