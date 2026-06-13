from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from app.core.logger import get_logger
from app.schemas import QueryRequest
from app.service.chatService import basicChatUsingGraph

load_dotenv()
logger = get_logger(__name__)
chatRouter = APIRouter()


@chatRouter.post(
    "/chat/completion",
    status_code=200,
    tags=["chat"],
)
async def chat_completion(payload: QueryRequest):

    return StreamingResponse(
        basicChatUsingGraph(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )