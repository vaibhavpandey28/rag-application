from pydantic import BaseModel, Field, field_validator
from typing import List,Optional


class QueryRequest(BaseModel):
    query: str
    # file: Optional[str] = Field(
    #     default=None, 
    #     description="Optional file name to query against. If not provided, the system will search across all indexed documents."
    # )
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



