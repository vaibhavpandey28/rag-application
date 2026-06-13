from pathlib import Path
import hashlib
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from app.core.logger import get_logger
from app.helpers.exceptionHandler import setup_exception_handlers

from app.schemas import QueryRequest, Citation, QueryResponse, UploadResponse
from app.routes.chatRoute import chatRouter
from app.llm import (rag, embeddings, llm)


load_dotenv()
logger = get_logger(__name__)

app = FastAPI(
    title="Basic  chat application with rag model",
    version="1.0.0"
)

setup_exception_handlers(app)

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


app.include_router(chatRouter, prefix="/api/v1", tags=["chat"])

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

