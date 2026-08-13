"""Shared state for the ReAct research agent."""

from __future__ import annotations

from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from operator import add


class SourceDoc(TypedDict, total=False):
    title: str
    url: str
    snippet: str
    text: str


class AgentState(TypedDict, total=False):
    topic: str
    max_steps: int
    step: int
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sources: Annotated[list[SourceDoc], add]
    status: Annotated[list[str], add]
    report: str


def initial_state(topic: str, max_steps: int = 8) -> dict[str, Any]:
    return {
        "topic": topic.strip(),
        "max_steps": max_steps,
        "step": 0,
        "messages": [],
        "sources": [],
        "status": [],
        "report": "",
    }
