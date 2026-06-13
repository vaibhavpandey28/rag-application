from dotenv import load_dotenv
from typing import Annotated, TypedDict,List
from typing_extensions import NotRequired
import json
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from app.schemas import QueryRequest
from app.core.logger import get_logger
from app.llm import llm, rag

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
    route: NotRequired[str]
    context: NotRequired[str]


# =====================================================
# Router Node
# =====================================================

def router(state: State):
    question = state["messages"][-1].content
    logger.info(question)
    routing_prompt = f""" You are a routing assistant. 
                        Determine whether the following question requires
                        document retrieval from a RAG system.
                        Question: {question}
                        Return ONLY ONE WORD: rag or chat
                    """

    response  = llm.invoke(routing_prompt)
    if isinstance(response,str):
        content = response
    else:
        content = response.content

    route = content.strip().lower()
    if "rag" in route:
        route = "rag"
    else:
        route = "chat"
    logger.info(f"Route selected: {route}")
    return { "route": route }



def route_query(state: State) -> str:
    return state.get("route", "chat")


# =====================================================
# Chat Node
# =====================================================

def chat_node(state: State):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm.invoke(messages)
    return {
        "messages": [response]
    }


# =====================================================
# RAG Retrieval Node
# =====================================================

def rag_node(state: State):
    question = state["messages"][-1].content
    result = rag.ask_query(question)

    context = result.get("answer", "")

    logger.info("RAG lookup completed")

    return {
        "context": context
    }


# =====================================================
# RAG Answer Node
# =====================================================

def rag_answer_node(state: State):

    question = state["messages"][-1].content

    context = state.get("context", "")

    messages = [
        SystemMessage(
            content=f""" You are Liza💜. Answer using ONLY the provided context. Context: {context} If the answer is not available in the context,say that you could not find it."""
        ),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# =====================================================
# Build Graph Once
# =====================================================

graph_builder = StateGraph(State)

graph_builder.add_node("router", router)
graph_builder.add_node("chat", chat_node)
graph_builder.add_node("rag", rag_node)
graph_builder.add_node("rag_answer", rag_answer_node)

graph_builder.add_edge(START, "router")

graph_builder.add_conditional_edges(
    "router",
    route_query,
    {
        "chat": "chat",
        "rag": "rag",
    },
)

graph_builder.add_edge("chat", END)

graph_builder.add_edge("rag", "rag_answer")
graph_builder.add_edge("rag_answer", END)

graph = graph_builder.compile()


# =====================================================
# FastAPI Streaming Function
# =====================================================

async def basicChatUsingGraph(payload: QueryRequest):

    """
    Handles a basic stateful chat session using a LangGraph workflow.
    Maintains conversation history using an in-memory checkpointer.
    """

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=payload.query)
            ]
        }
    )

    final_response = result["messages"][-1].content
    logger.info("result>>",result)

    yield (
        f"data: {json.dumps({'content': final_response})}\n\n"
    )

    yield (
        f"data: {json.dumps({'done': True})}\n\n"
    )