from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import asyncio
from langchain_community.callbacks import BaseCallbackHandler
from app.core.logger import get_logger

logger = get_logger(__name__)

DB_NAME = "./chroma_db"
COLLECTION_NAME = "documents"


class RAGService:
    """Simplified RAG service: index documents and ask queries.

    This version removes streaming and callback complexity and keeps
    two straightforward methods: `index_documents` and `ask_query`.
    """

    def __init__(self, embeddings, llm):
        self.embeddings = embeddings
        self.llm = llm

    def _get_loader(self, file_path: str):
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return PyMuPDFLoader(file_path)
        if suffix in (".txt", ".text"):
            return TextLoader(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def index_documents(self, file_path: str, source_id: str | None = None, source_name: str | None = None):
        loader = self._get_loader(file_path)
        documents = loader.load()
        file_name = source_name or Path(file_path).name
        for doc in documents:
            if not hasattr(doc, "metadata") or doc.metadata is None:
                doc.metadata = {}
            doc.metadata["source"] = file_name
            if source_id is not None:
                doc.metadata["source_id"] = source_id

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        logger.info("Indexing %d chunks into Chroma DB", len(chunks))
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_NAME,
        )
        return {"chunks": len(chunks), "source": file_name}

    def ask_query(self, query: str, top_k: int = 5):
        db = Chroma(collection_name=COLLECTION_NAME, persist_directory=DB_NAME, embedding_function=self.embeddings)
        docs = db.similarity_search(query=query, k=top_k)
        logger.info("Retrieved %d chunks for query", len(docs))

        context = "\n\n".join(getattr(doc, "page_content", "") for doc in docs)
        prompt = f"You are a helpful AI assistant. Answer only from the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Call the LLM synchronously and return a simple structure.
        resp = self.llm.invoke(prompt)

        citations = []
        seen = set()
        for idx, doc in enumerate(docs, start=1):
            src = (getattr(doc, "metadata", {}) or {}).get("source", "unknown")
            citations.append({"rank": idx, "source": src, "text": getattr(doc, "page_content", "")})
            seen.add(src)

        return {"answer": getattr(resp, "content", str(resp)), "sources": list(seen), "citations": citations}

    def ask_query_stream(self, query: str, queue: "asyncio.Queue", top_k: int = 5):
        """Retrieve context and stream LLM tokens into `queue`.

        Tries known LangChain callback patterns (`generate`, `predict`, `invoke` with
        `callbacks` param). If callbacks aren't supported, falls back to calling
        `invoke` and streaming the full response by words.
        """
        db = Chroma(collection_name=COLLECTION_NAME, persist_directory=DB_NAME, embedding_function=self.embeddings)
        docs = db.similarity_search(query=query, k=top_k)

        context = "\n\n".join(getattr(doc, "page_content", "") for doc in docs)
        prompt = f"You are a helpful AI assistant. Answer only from the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        class QueueCallbackHandler(BaseCallbackHandler):
            def __init__(self, queue: "asyncio.Queue"):
                self.queue = queue

            def on_llm_new_token(self, token: str, **kwargs):
                try:
                    self.queue.put_nowait(token)
                except Exception:
                    pass

            def on_llm_end(self, **kwargs):
                try:
                    self.queue.put_nowait(None)
                except Exception:
                    pass

        handler = QueueCallbackHandler(queue)

        # Try to use callback-capable methods first
        try:
            if hasattr(self.llm, "generate"):
                self.llm.generate([prompt], callbacks=[handler])
                return
            if hasattr(self.llm, "predict"):
                self.llm.predict(prompt, callbacks=[handler])
                return
            if hasattr(self.llm, "invoke"):
                try:
                    # Some LLM wrappers accept callbacks on invoke
                    self.llm.invoke(prompt, callbacks=[handler])
                    return
                except TypeError:
                    pass
        except Exception:
            try:
                queue.put_nowait(None)
            except Exception:
                pass

        # Fallback: synchronous invoke and stream by words
        try:
            resp = self.llm.invoke(prompt)
            content = getattr(resp, "content", str(resp))
            # Stream token-like chunks (words) for simplicity
            for token in content.split():
                try:
                    queue.put_nowait(token + " ")
                except Exception:
                    pass
            try:
                queue.put_nowait(None)
            except Exception:
                pass
        except Exception as e:
            try:
                queue.put_nowait(f"[error] {e}")
                queue.put_nowait(None)
            except Exception:
                pass