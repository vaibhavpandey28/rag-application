from pathlib import Path
import hashlib
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import asyncio
from pydantic import BaseModel, SecretStr, field_validator
from typing import List

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.rag import RAGService
from app.core.logger import get_logger
from app.helpers.exceptionHandler import setup_exception_handlers

load_dotenv()

logger = get_logger(__name__)

app = FastAPI(
    title="Basic RAG Pipeline API",
    version="1.0.0"

)

setup_exception_handlers(app)

API_KEY = SecretStr(os.getenv("OPENAI_API_KEY", "sk-not-needed"))
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_MANIFEST = UPLOAD_DIR / "manifest.json"


def load_upload_manifest() -> dict:
    if not UPLOAD_MANIFEST.exists():
        return {}
    try:
        return json.loads(UPLOAD_MANIFEST.read_text())
    except json.JSONDecodeError:
        return {}


def save_upload_manifest(manifest: dict) -> None:
    UPLOAD_MANIFEST.write_text(json.dumps(manifest, indent=2))

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) 


# Chat Model
llm = ChatOpenAI(
    model="google/gemma-4-e2b",
    base_url="http://127.0.0.1:2402/v1",
    api_key=API_KEY,
    temperature=0,
)

rag = RAGService(
    embeddings=embeddings,
    llm=llm,
)


class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Query cannot be empty")
        return value


class Citation(BaseModel):
    rank: int
    source: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    citations: List[Citation]


class UploadResponse(BaseModel):
    message: str
    chunks: int
    source: str


@app.get("/")
def root():
    return {"message": "RAG API Running"}


@app.post("/upload/file", response_model=UploadResponse)
def upload_file(file: UploadFile = File(...)):

    data = file.file.read()
    file_hash = hashlib.sha256(data).hexdigest()

    manifest = load_upload_manifest()
    if file_hash in manifest:
        return {
            "message": "Document already indexed",
            "chunks": manifest[file_hash]["chunks"],
            "source": manifest[file_hash]["source"],
        }

    filename = Path(file.filename or "uploaded_file").name
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed_extensions = {"pdf", "txt", "text"}
    if extension not in allowed_extensions:
        raise ValueError("Unsupported file type. Supported extensions are: .pdf, .txt")

    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path = UPLOAD_DIR / f"{file_hash[:8]}_{filename}"

    with open(file_path, "wb") as f:
        f.write(data)

    result = rag.index_documents(str(file_path), source_id=file_hash, source_name=filename)

    manifest[file_hash] = {
        "source": filename,
        "path": str(file_path),
        "chunks": result["chunks"],
    }
    save_upload_manifest(manifest)

    return {
        "message": "Document indexed successfully",
        **result,
    }


@app.post("/query", response_model=QueryResponse)
def query_document(payload: QueryRequest):
    result = rag.ask_query(payload.query)
    return result


@app.get("/query/stream")
async def stream_query(q: str):
    queue: asyncio.Queue = asyncio.Queue()
    async def run_model():
        await asyncio.to_thread(rag.ask_query_stream, q, queue)

    asyncio.create_task(run_model())

    async def event_generator():
        while True:
            token = await queue.get()
            if token is None:
                break
            yield f"data: {token}\n\n"
        yield "event: done\ndata: \n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")