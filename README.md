# Basic RAG Pipeline

A Retrieval-Augmented Generation (RAG) application built with FastAPI, LangGraph, ChromaDB, HuggingFace Embeddings, and Langfuse Observability.

## Features

* FastAPI REST API
* LangGraph workflow orchestration
* Retrieval-Augmented Generation (RAG)
* ChromaDB vector storage
* HuggingFace sentence embeddings
* DeepSeek/OpenAI-compatible LLM support
* Langfuse tracing and observability
* Stateful conversation memory using LangGraph MemorySaver
* Streaming responses via Server-Sent Events (SSE)

---

## Architecture

```text
User Query
    │
    ▼
FastAPI Endpoint
    │
    ▼
LangGraph Workflow
    │
    ├── Chat Node
    │
    └── RAG Node
           │
           ▼
      ChromaDB
           │
           ▼
      Retrieved Context
           │
           ▼
          LLM
           │
           ▼
      Final Response
```

---

## Tech Stack

* Python 3.13+
* FastAPI
* LangGraph
* LangChain
* ChromaDB
* HuggingFace Embeddings
* DeepSeek R1
* Langfuse
* Uvicorn

---

## Project Structure

```text
app/
│
├── main.py
├── rag.py
│
├── routes/
│   └── chatRoute.py
│
├── service/
│   └── chatService.py
│
├── graph/
│   └── graph.py
│
├── models/
│   └── schemas.py
│
├── utils/
│
└── vectorstore/
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key

HUGGINGFACEHUB_API_TOKEN=your_hf_token

LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd basic-rag-pipeline
```

### Create Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## Run Application

```bash
uvicorn app.main:app --reload
```

Application will start at:

```text
http://127.0.0.1:8000
```

---

## API Endpoint

### Chat Completion

```http
POST /api/v1/chat/completion
```

Request:

```json
{
  "query": "What is LangGraph?"
}
```

Response:

```json
{
  "content": "LangGraph is..."
}
```

---

## LangGraph Memory

Conversation state is maintained using:

```python
memory = MemorySaver()

graph = graph_builder.compile(
    checkpointer=memory
)
```

Each request must include a unique:

```python
config = {
    "configurable": {
        "thread_id": "unique-chat-id"
    }
}
```

---

## Langfuse Tracing

Tracing is enabled using:

```python
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()
```

Graph execution:

```python
await graph.ainvoke(
    inputs,
    config={
        "callbacks": [langfuse_handler]
    }
)
```

---

## Embeddings Model

```python
sentence-transformers/all-MiniLM-L6-v2
```

Used for document chunk embedding and retrieval.

---

## Supported Models

Example:

```python
ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1",
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)
```

Can be replaced with:

* GPT-4o
* GPT-5
* DeepSeek
* Gemma
* Llama
* Mistral

---

## Future Improvements

* Human-in-the-loop approval
* Persistent memory storage
* Authentication
* Multi-user sessions
* Streaming token responses
* Evaluation and feedback loops
* Agentic workflows

---

## Author

Vaibhav Pandey

Built using FastAPI, LangGraph, ChromaDB, HuggingFace, and Langfuse.
