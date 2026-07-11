"""
Streamlit UI for the GigaCorp Customer Support RAG Agent.

Session/memory handling: Streamlit re-runs this whole script on every user
interaction, so `st.session_state` (which persists across re-runs within one
browser session) is what carries the chat history and the loaded vector
store forward -- this IS the "session memory" the assignment asks about.

UI design notes: the visual language is built around the one thing that
makes this agent's answers verifiable rather than a black box -- exact
line-number citations. Sources render as small "citation chips" (amber
marker + mono file:line reference) rather than generic cards with a
fabricated "confidence score" -- the backend returns a rank and a line
range, not a similarity score, so the UI doesn't imply data it doesn't have.

Rendering note: every raw HTML/CSS fragment in this file goes through
st.html(), not st.markdown(unsafe_allow_html=True). st.markdown() still
runs the string through Streamlit's Markdown parser first, which can
mangle a large <style> block (part of it gets treated as plain text
instead of CSS). st.html() injects the string directly with no Markdown
pass, so styles apply reliably.
"""

import time

import streamlit as st
from dotenv import load_dotenv

from src.vectorstore import load_vectorstore
from src.rag_chain import answer_question

load_dotenv()

st.set_page_config(
    page_title="GigaCorp Support",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_LABEL = "Llama 3.1 8B · Groq"

# ──────────────────────────────────────────────────────────────────────────
# Theme (single st.html call -- real <style>, not markdown-parsed text)
# ──────────────────────────────────────────────────────────────────────────
THEME = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0B0C10;
  --surface: #15171C;
  --surface-hover: #1B1E25;
  --border: #262A33;
  --text: #EDEDF0;
  --text-dim: #8B8F9C;
  --text-faint: #6B707C;
  --accent: #E8A33D;
  --accent-dim: rgba(232, 163, 61, 0.12);
  --online: #34D399;
  --font-display: 'Space Grotesk', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}
html, body, [class*="css"] { font-family: var(--font-body); }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 820px; padding-top: 2.2rem; padding-bottom: 7rem; }
*:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
section[data-testid="stSidebar"] hr { border-color: var(--border); margin: 1.1rem 0; }
section[data-testid="stSidebar"] .stButton button {
  background: transparent; border: 1px solid var(--border); color: var(--text-dim);
  border-radius: 8px; font-size: 13px; font-family: var(--font-body); font-weight: 500;
  padding: 0.45rem 0.8rem; width: 100%; transition: all .15s ease;
}
section[data-testid="stSidebar"] .stButton button:hover {
  border-color: var(--accent); color: var(--accent); background: var(--accent-dim);
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--border); border-radius: 8px; background: transparent;
}
section[data-testid="stSidebar"] summary { font-size: 13px; color: var(--text-dim); }
section[data-testid="stSidebar"] [data-testid="stExpander"] p { font-size: 12.5px; color: var(--text-faint); line-height: 1.6; }

.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.brand-mark {
  width: 28px; height: 28px; border-radius: 7px; background: var(--accent); color: #0B0C10;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--font-display); font-weight: 700; font-size: 14px; flex-shrink: 0;
}
.brand-name { font-family: var(--font-display); font-weight: 600; font-size: 15px; color: var(--text); }
.brand-sub { font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); margin-left: 38px; }

.side-label {
  font-family: var(--font-mono); font-size: 10.5px; letter-spacing: .08em;
  color: var(--text-faint); text-transform: uppercase; margin: 0 0 8px 2px;
}
.stat-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 2px; font-size: 13px; color: var(--text-dim); border-bottom: 1px solid var(--border);
}
.stat-row:last-child { border-bottom: none; }
.stat-row .val { font-family: var(--font-mono); color: var(--text); font-size: 12.5px; }

.status-pill { display: inline-flex; align-items: center; gap: 6px; font-family: var(--font-mono); font-size: 11.5px; color: var(--text-dim); }
.dot { width: 6px; height: 6px; border-radius: 50%; background: var(--online); box-shadow: 0 0 6px var(--online); }

/* Top bar */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 1.6rem; margin-bottom: 1.6rem; border-bottom: 1px solid var(--border);
}
.topbar-title { font-family: var(--font-display); font-weight: 600; font-size: 17px; color: var(--text); }
.topbar-badge {
  font-family: var(--font-mono); font-size: 11px; color: var(--text-faint);
  border: 1px solid var(--border); padding: 3px 9px; border-radius: 20px;
}

/* Empty state */
.empty-wrap { text-align: center; padding: 3.2rem 0 1.6rem; }
.empty-mark {
  width: 46px; height: 46px; border-radius: 12px; background: var(--accent-dim);
  border: 1px solid var(--accent); color: var(--accent); font-family: var(--font-display);
  font-weight: 700; font-size: 20px; display: flex; align-items: center; justify-content: center;
  margin: 0 auto 18px;
}
.empty-title { font-family: var(--font-display); font-weight: 600; font-size: 26px; color: var(--text); margin-bottom: 8px; }
.empty-sub { font-size: 14px; color: var(--text-dim); max-width: 420px; margin: 0 auto; line-height: 1.6; }

/* Suggestion chips (native st.button, restyled) */
div[data-testid="column"] .stButton button {
  background: var(--surface); border: 1px solid var(--border); color: var(--text-dim);
  border-radius: 10px; font-size: 12.5px; font-family: var(--font-body);
  padding: 0.7rem 0.6rem; text-align: left; width: 100%; transition: all .15s ease;
}
div[data-testid="column"] .stButton button:hover {
  border-color: var(--accent); color: var(--text); background: var(--accent-dim);
  transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

/* Chat messages */
@keyframes fadeIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: translateY(0); } }
[data-testid="stChatMessage"] {
  background: transparent; border: none; padding: 0.9rem 0; gap: 12px;
  border-bottom: 1px solid var(--border); animation: fadeIn .25s ease;
}
[data-testid="stChatMessageAvatarUser"] { background: var(--border) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent) !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 14.5px; line-height: 1.65; color: var(--text); }
[data-testid="stChatMessage"] code {
  font-family: var(--font-mono); background: var(--surface); color: var(--accent);
  padding: 1px 5px; border-radius: 4px; font-size: 13px;
}

/* Citation chips */
.cite-label {
  font-family: var(--font-mono); font-size: 10.5px; letter-spacing: .06em;
  color: var(--text-faint); text-transform: uppercase; margin: 10px 0 6px 2px;
}
.cite-row { display: flex; flex-wrap: wrap; gap: 8px; }
.cite-chip {
  display: flex; align-items: center; gap: 7px; background: var(--surface);
  border: 1px solid var(--border); border-left: 2px solid var(--accent);
  border-radius: 6px; padding: 6px 10px; font-size: 12px; transition: border-color .15s ease;
}
.cite-chip:hover { border-color: var(--accent); }
.cite-chip .n { font-family: var(--font-mono); color: var(--accent); font-size: 11px; }
.cite-chip .sec { color: var(--text-dim); font-weight: 500; }
.cite-chip .loc { font-family: var(--font-mono); color: var(--text-faint); font-size: 11px; }

.retrieval-line {
  font-family: var(--font-mono); font-size: 12px; color: var(--text-faint);
  display: flex; align-items: center; gap: 8px; padding: 4px 0;
}

/* Chat input */
[data-testid="stChatInput"] { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-size: 14px; }
</style>
"""
st.html(THEME)

# ──────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = load_vectorstore()
if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

chunk_count = st.session_state.vectorstore.index.ntotal
turn_count = len(st.session_state.messages) // 2

# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.html(
        '<div class="brand"><div class="brand-mark">G</div>'
        '<div class="brand-name">GigaCorp Support</div></div>'
        '<div class="brand-sub">RAG customer support agent</div>'
    )

    st.html('<div style="margin-top:18px"></div>')
    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_latency = None
        st.rerun()

    st.html("<hr>")
    st.html('<div class="side-label">Knowledge base</div>')
    st.html(f"""
    <div class="stat-row"><span>Indexed chunks</span><span class="val">{chunk_count}</span></div>
    <div class="stat-row"><span>Vector store</span><span class="val">FAISS</span></div>
    <div class="stat-row"><span>Embeddings</span><span class="val">MiniLM-L6-v2</span></div>
    <div class="stat-row"><span>Retrieval</span><span class="val">top-k similarity</span></div>
    """)

    st.html("<hr>")
    st.html('<div class="side-label">Session</div>')
    latency_row = (
        f'<div class="stat-row"><span>Last response</span><span class="val">{st.session_state.last_latency:.1f}s</span></div>'
        if st.session_state.last_latency else ""
    )
    st.html(f"""
    <div class="stat-row"><span>Model</span><span class="val">{MODEL_LABEL}</span></div>
    <div class="stat-row"><span>Memory</span><span class="val">last 3 turns</span></div>
    <div class="stat-row"><span>Turns this session</span><span class="val">{turn_count}</span></div>
    {latency_row}
    """)

    st.html("<hr>")
    st.html('<span class="status-pill"><span class="dot"></span>Online</span>')

    st.html("<hr>")
    with st.expander("About this agent"):
        st.markdown(
            "Answers are grounded only in GigaCorp's local FAQ document, "
            "retrieved via FAISS similarity search. Every claim is traced "
            "back to an exact source line range — nothing is answered from "
            "outside knowledge."
        )

# ──────────────────────────────────────────────────────────────────────────
# Top bar
# ──────────────────────────────────────────────────────────────────────────
st.html(f"""
<div class="topbar">
    <div class="topbar-title">Customer Support</div>
    <div class="topbar-badge">{MODEL_LABEL}</div>
</div>
""")

SUGGESTIONS = [
    "What's your shipping cost to India?",
    "How do I return an item?",
    "What are your business hours?",
    "What's included in the Premium tier?",
    "What payment methods do you accept?",
]

# ──────────────────────────────────────────────────────────────────────────
# Empty state
# ──────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.html("""
    <div class="empty-wrap">
        <div class="empty-mark">G</div>
        <div class="empty-title">How can I help?</div>
        <div class="empty-sub">Ask about shipping, returns, business hours, service tiers,
        or account &amp; payment — every answer is cited to an exact line in GigaCorp's FAQ.</div>
    </div>
    """)

    cols = st.columns(len(SUGGESTIONS))
    picked = None
    for col, q in zip(cols, SUGGESTIONS):
        with col:
            if st.button(q, use_container_width=True, key=f"sugg_{q}"):
                picked = q
    if picked:
        st.session_state.pending_input = picked
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def render_sources(sources):
    if not sources:
        return
    st.html('<div class="cite-label">Sources</div>')
    chips = "".join(
        f'<div class="cite-chip"><span class="n">[{i}]</span>'
        f'<span class="sec">{s["section"].title()}</span>'
        f'<span class="loc">{s["source"]} · L{s["line_start"]}-{s["line_end"]}</span></div>'
        for i, s in enumerate(sources, start=1)
    )
    st.html(f'<div class="cite-row">{chips}</div>')


def run_turn(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with placeholder:
            st.html('<span class="retrieval-line">⟶ searching knowledge base…</span>')
        start = time.time()
        answer, sources = answer_question(
            st.session_state.vectorstore,
            user_input,
            st.session_state.messages[:-1],
        )
        elapsed = time.time() - start
        st.session_state.last_latency = elapsed
        placeholder.empty()
        st.markdown(answer)
        render_sources(sources)
        st.caption(f"Answered in {elapsed:.1f}s")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


# ──────────────────────────────────────────────────────────────────────────
# Chat history
# ──────────────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# ──────────────────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────────────────
if "pending_input" in st.session_state:
    pending = st.session_state.pop("pending_input")
    run_turn(pending)
    st.rerun()

user_input = st.chat_input("Ask about shipping, returns, hours, tiers…")
if user_input:
    run_turn(user_input)
    st.rerun()