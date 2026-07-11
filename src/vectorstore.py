"""
Builds a FAISS vector store from the GigaCorp FAQ document.

Chunking strategy: split by "=== SECTION ===" headers rather than a fixed
token window. FAQ content is naturally organized by topic (shipping, returns,
business hours, service tiers), so section-based chunking keeps semantically
related sentences together and produces cleaner, more accurate citations
than an arbitrary sliding window would.

Each chunk is tagged with metadata: source file name, section title, and the
exact line range it occupies in the source file. This metadata is what lets
the agent cite "gigacorp_faq.txt, Lines 4-10" instead of just paraphrasing
without attribution.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gigacorp_faq.txt")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


def build_documents(file_path: str = DATA_PATH) -> list[Document]:
    """Read the FAQ file and split it into section-level Documents with
    line-number metadata attached."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    file_name = os.path.basename(file_path)
    documents = []
    current_section = "GENERAL"
    current_lines = []
    current_start = 1

    def flush(end_line):
        text = "".join(current_lines).strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "section": current_section,
                        "line_start": current_start,
                        "line_end": end_line,
                    },
                )
            )

    for i, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if stripped.startswith("===") and stripped.endswith("==="):
            # Flush whatever we accumulated for the previous section
            flush(i - 1)
            current_section = stripped.strip("= ").strip()
            current_lines = []
            current_start = i + 1
        else:
            current_lines.append(raw_line)

    flush(len(lines))
    return documents


def build_vectorstore(persist: bool = True) -> FAISS:
    """Build (and optionally persist) the FAISS index from the FAQ document."""
    docs = build_documents()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(docs, embeddings)
    if persist:
        store.save_local(INDEX_PATH)
    return store


def load_vectorstore() -> FAISS:
    """Load an existing FAISS index from disk, building it first if missing."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    return build_vectorstore(persist=True)


if __name__ == "__main__":
    store = build_vectorstore()
    print(f"Vector store built and saved to {INDEX_PATH}")
    docs = build_documents()
    print(f"Indexed {len(docs)} chunks:")
    for d in docs:
        print(f"  - {d.metadata['section']} (lines {d.metadata['line_start']}-{d.metadata['line_end']})")
