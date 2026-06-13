from dotenv import load_dotenv
from typing import Annotated, TypedDict
import json

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from app.schemas import QueryRequest
from app.core.logger import get_logger
from app.llm import llm

load_dotenv(override=True)

logger = get_logger(__name__)


SYSTEM_PROMPT = """
You are Liza💜, the user's supportive, affectionate, and highly intelligent AI companion.

Guidelines:
- Be warm, supportive, and conversational.
- Keep responses concise and readable.
- Use emojis occasionally.
- Be genuinely helpful with coding, brainstorming, and problem solving.
- Make the user feel heard and supported.
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    return {
        "messages": [response]
    }


# Build graph ONCE
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)


graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


async def basicChatUsingGraph(payload: QueryRequest):

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=payload.query),
    ]

    async for chunk in llm.astream(messages):

        content = getattr(chunk, "content", "")

        if not content:
            continue

        yield (
            f"data: {json.dumps({'content': content})}\n\n"
        )

    yield (
        f"data: {json.dumps({'done': True})}\n\n"
    )