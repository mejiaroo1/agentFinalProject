"""Shared state for the multi-agent research team."""

from __future__ import annotations

from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from operator import add

from scout.state import SourceDoc


class TeamState(TypedDict, total=False):
    topic: str
    max_steps: int
    max_rounds: int
    step: int
    round: int
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sources: Annotated[list[SourceDoc], add]
    status: Annotated[list[str], add]
    plan: str
    draft: str
    critique: str
    verdict: str  # "ok" | "retry"
    report: str


def initial_team_state(
    topic: str,
    max_steps: int = 8,
    max_rounds: int = 2,
) -> dict[str, Any]:
    return {
        "topic": topic.strip(),
        "max_steps": max_steps,
        "max_rounds": max_rounds,
        "step": 0,
        "round": 0,
        "messages": [],
        "sources": [],
        "status": [],
        "plan": "",
        "draft": "",
        "critique": "",
        "verdict": "",
        "report": "",
    }
