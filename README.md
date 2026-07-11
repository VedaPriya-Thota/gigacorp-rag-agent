# GigaCorp Customer Support RAG Agent

A conversational customer support assistant that answers questions using a local knowledge base (RAG), cites its sources by line number, and remembers context across a conversation.

🔗 **Live App:** [gigacorp-rag-agentgit-cnjkiosukttf8eahvwvwz5.streamlit.app](https://gigacorp-rag-agentgit-cnjkiosukttf8eahvwvwz5.streamlit.app)

🎥 **Demo Video:** [Watch the walkthrough](https://drive.google.com/file/d/1i5eErSTRqlHLDEv7YHcfztT2VmKIP_fP/view?usp=sharing) — covers user flow, frontend, backend calls, vector DB retrieval, and session memory.

## Architecture

```
User (browser)
   │
   ▼
Streamlit UI (app.py)
   │  - renders chat, sources panel, session state (memory)
   ▼
RAG Chain (src/rag_chain.py)
   │  - formats prompt with retrieved context + chat history
   │  - calls Groq LLM (Llama 3.1 8B Instant)
   ▼
Vector Store (src/vectorstore.py)
   │  - FAISS index over data/gigacorp_faq.txt
   │  - chunked by FAQ section, tagged with line-number metadata
   ▼
Local Embeddings (sentence-transformers/all-MiniLM-L6-v2, forced to CPU device)
```

**Why this stack:**
- **FAISS + local HuggingFace embeddings** — no API cost or key required just to embed the knowledge base; only the chat LLM call needs a key.
- **Groq (Llama 3.1 8B Instant)** — free tier, very low latency, good enough reasoning for FAQ-style Q&A.
- **Section-based chunking** (not fixed-size sliding windows) — the FAQ is naturally organized by topic (shipping, returns, hours, tiers), so chunking by section keeps each citation coherent and topically complete.
- **Explicit history-in-prompt memory** — rather than an opaque memory object, the last few turns are passed directly into the prompt. This is what lets the agent resolve "how much to ship *there*?" back to a country mentioned two turns earlier, and it's easy to inspect/debug.
- **CPU-forced embeddings device** — explicitly set via `model_kwargs={"device": "cpu"}` to avoid a known `NotImplementedError: Cannot copy out of meta tensor` crash caused by newer `transformers`/`accelerate` versions defaulting to lazy meta-device model loading on hosted platforms like Streamlit Cloud.

## Features

- Chat interface with persistent session memory (survives across turns, resets on "Clear conversation")
- Retrieval-augmented answers grounded only in the local FAQ document
- Explicit citations: every claim is tagged `[Source: gigacorp_faq.txt, Lines X-Y]`, and a collapsible "Sources" panel shows exactly which FAQ section was used
- Handles multi-turn context (e.g., "Do you ship to India?" → "Yes." → "How much does it cost?")

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd gigacorp-rag-agent
   pip install -r requirements.txt
   ```

2. **Set your API key**
   ```bash
   cp .env.example .env
   # then edit .env and add your GROQ_API_KEY
   # get a free key at https://console.groq.com/keys
   ```

3. **(Optional) Pre-build the vector index**
   ```bash
   python -m src.vectorstore
   ```
   This isn't required — the app builds the index automatically on first run if it doesn't exist — but running it once locally is a good sanity check that everything loads correctly.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub (make sure `.env` is **not** committed — it's already in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub repo, and set the main file to `app.py`.
3. In the app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy. First load will take a little longer as the embedding model downloads.

**Known deployment gotcha:** if you see `NotImplementedError: Cannot copy out of meta tensor` on first deploy, it means `transformers`/`huggingface-hub` resolved to versions that changed their default model-loading behavior. This repo's `requirements.txt` already pins compatible versions (`transformers==4.44.2`, `huggingface-hub==0.24.6`, `tokenizers==0.19.1`) and `vectorstore.py` explicitly forces `device="cpu"` to avoid it — if you fork this repo, keep those pins.

## Folder Structure

```
gigacorp-rag-agent/
├── app.py                 # Streamlit UI, session state / memory
├── data/
│   └── gigacorp_faq.txt   # mock knowledge base
├── src/
│   ├── vectorstore.py     # chunking + FAISS index build/load
│   └── rag_chain.py       # retrieval, prompt, LLM call
├── requirements.txt
├── .env.example
└── README.md
```

## Test Instructions

No login required. Try it directly on the [live app](https://gigacorp-rag-agentgit-cnjkiosukttf8eahvwvwz5.streamlit.app), or run locally. Suggested test flow to demonstrate memory + citations:

1. Ask: **"Do you ship to India?"**
2. Then ask: **"How much does it cost to ship there?"** — the agent should resolve "there" to India without you repeating it.
3. Ask: **"Can I return a final sale item?"** — check the Sources panel cites the Return Policy section.

## Future Improvements

- Streaming token-by-token responses instead of waiting for the full answer
- Swap in a persistent memory store (e.g., SQLite) so history survives a page refresh, not just a session
- Support uploading a custom FAQ document instead of a hardcoded one