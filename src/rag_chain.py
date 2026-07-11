"""
Core RAG logic: retrieval + citation-aware prompting + conversational memory.

Memory design: rather than pulling in a heavyweight memory abstraction, we
pass the last N turns of chat history directly into the prompt as plain text.
This is intentional -- for a single-session customer support chat, explicit
history-in-context is more transparent and debuggable than an opaque memory
object, and it's exactly what lets the model resolve references like "how
much does it cost to ship THERE" back to "India" mentioned two turns earlier.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import load_vectorstore

# Load environment variables from .env file
load_dotenv()

SYSTEM_PROMPT = """You are GigaCorp's customer support assistant. Answer the customer's question using ONLY the information in the CONTEXT below. Do not use outside knowledge.

Rules:
1. Every factual claim must be followed by a citation in this exact format: [Source: <filename>, Lines <start>-<end>]
2. If the answer isn't in the CONTEXT, say you don't have that information and suggest contacting support directly. Do not make anything up.
3. Use the CONVERSATION HISTORY to resolve references like "there", "it", or "that" to what the customer actually means.
4. Keep answers concise and friendly, like a real support agent.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

CUSTOMER QUESTION:
{question}
"""


def get_llm():
    """Groq's Llama 3.1 -- chosen for free-tier availability and low latency,
    which matters for a chat UI where users expect near-instant replies."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


def format_history(history: list[dict]) -> str:
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history[-6:]:  # last 3 exchanges is plenty of context for a FAQ bot
        role = "Customer" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def retrieve_context(vectorstore, query: str, k: int = 3) -> tuple[str, list[dict]]:
    """Retrieve top-k relevant chunks and format them with citation metadata
    inline, so the LLM has the exact source/line info available to quote."""
    results = vectorstore.similarity_search(query, k=k)
    context_blocks = []
    sources = []
    for doc in results:
        meta = doc.metadata
        tag = f"[Source: {meta['source']}, Lines {meta['line_start']}-{meta['line_end']}]"
        context_blocks.append(f"{tag}\n{doc.page_content}")
        sources.append(meta)
    return "\n\n".join(context_blocks), sources


def answer_question(vectorstore, question: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Run one full RAG turn: retrieve -> build prompt -> call LLM -> return
    (answer, sources_used) so the UI can render a separate sources panel."""
    context, sources = retrieve_context(vectorstore, question)
    prompt = SYSTEM_PROMPT.format(
        context=context,
        history=format_history(history),
        question=question,
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content, sources
