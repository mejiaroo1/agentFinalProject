"""Multi-agent team: Planner → Researcher↔tools → Writer → Critic (loop)."""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from replab.llm import get_api_key, openai_model, require_api_key
from scout.tools import AGENT_TOOLS, sources_from_tool_content
from scout_team.prompts import (
    CRITIC_SYSTEM,
    PLANNER_SYSTEM,
    RESEARCHER_SYSTEM,
    WRITER_SYSTEM,
)
from scout_team.state import TeamState, initial_team_state

load_dotenv()

StatusCallback = Callable[[str], None]


def _cfg() -> dict[str, Any]:
    return {
        "api_key": get_api_key(),
        "model": openai_model(),
        "max_steps": int(os.getenv("MAX_AGENT_STEPS", os.getenv("MAX_LOOPS", "8"))),
        "max_rounds": int(os.getenv("MAX_TEAM_ROUNDS", "2")),
    }


def _require_api_key() -> str:
    return require_api_key()


def _llm() -> ChatOpenAI:
    c = _cfg()
    return ChatOpenAI(
        model=c["model"],
        api_key=_require_api_key(),
        temperature=0.2,
    )


def _text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        ]
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def planner_node(state: TeamState) -> dict[str, Any]:
    topic = state.get("topic", "")
    resp = _llm().invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Topic:\n{topic}"),
        ]
    )
    plan = _text(resp)
    return {
        "plan": plan,
        "status": ["Planner: research plan ready"],
        "messages": [
            SystemMessage(content=RESEARCHER_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic:\n{topic}\n\n## Research Plan\n{plan}\n\n"
                    "Investigate using tools, then write an evidence brief."
                )
            ),
        ],
    }


def researcher_node(state: TeamState) -> dict[str, Any]:
    step = int(state.get("step", 0)) + 1
    messages = list(state.get("messages") or [])
    max_steps = int(state.get("max_steps", 8))
    invoke_messages = list(messages)

    critique = (state.get("critique") or "").strip()
    verdict = (state.get("verdict") or "").strip().lower()
    if verdict == "retry" and critique and step == 1:
        # After a critic retry, reseed with critique guidance
        invoke_messages = messages + [
            HumanMessage(
                content=(
                    "The Critic requested more work:\n"
                    f"{critique}\n\n"
                    "Gather additional evidence with tools, then write an updated evidence brief."
                )
            )
        ]

    if step >= max_steps - 1:
        invoke_messages = invoke_messages + [
            HumanMessage(
                content=(
                    "You are near the research step limit. Do not call more tools. "
                    "Write the evidence brief now."
                )
            )
        ]

    resp = _llm().bind_tools(AGENT_TOOLS).invoke(invoke_messages)
    status = [f"Researcher step {step}"]
    tool_calls = getattr(resp, "tool_calls", None) or []
    if tool_calls:
        names = [tc.get("name", "tool") for tc in tool_calls]
        status.append("Researcher tools: " + ", ".join(names))
    else:
        status.append("Researcher: evidence brief ready")

    if not state.get("messages"):
        return {"step": step, "messages": messages + [resp], "status": status}
    # On critic-driven restart, messages may already exist; append response only
    if verdict == "retry" and step == 1 and critique:
        return {"step": step, "messages": [HumanMessage(content=f"Critic feedback:\n{critique}"), resp], "status": status}
    return {"step": step, "messages": [resp], "status": status}


def tools_node(state: TeamState) -> dict[str, Any]:
    result = ToolNode(AGENT_TOOLS).invoke(state)
    new_messages = result.get("messages") or []
    sources: list[dict[str, Any]] = []
    status: list[str] = []
    for msg in new_messages:
        if not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", "") or "tool"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        status.append(f"Tool result: {name}")
        sources.extend(sources_from_tool_content(name, content))
    out: dict[str, Any] = {"messages": new_messages, "status": status}
    if sources:
        out["sources"] = sources
    return out


def writer_node(state: TeamState) -> dict[str, Any]:
    topic = state.get("topic", "")
    plan = state.get("plan", "")
    transcript = list(state.get("messages") or [])
    critique = (state.get("critique") or "").strip()
    extra = ""
    if (state.get("verdict") or "").lower() == "retry" and critique:
        extra = f"\n\nAddress this critic feedback:\n{critique}\n"

    resp = _llm().invoke(
        [
            SystemMessage(content=WRITER_SYSTEM),
            *transcript,
            HumanMessage(
                content=(
                    f"Topic: {topic}\n\nPlan:\n{plan}\n"
                    f"{extra}\n"
                    "Write the full cited markdown report now."
                )
            ),
        ]
    )
    draft = _text(resp)
    return {
        "draft": draft,
        "status": ["Writer: draft report ready"],
    }


def critic_node(state: TeamState) -> dict[str, Any]:
    topic = state.get("topic", "")
    plan = state.get("plan", "")
    draft = state.get("draft", "")
    round_n = int(state.get("round", 0)) + 1

    resp = _llm().invoke(
        [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {topic}\n\n## Plan\n{plan}\n\n## Draft\n{draft}\n"
                )
            ),
        ]
    )
    critique = _text(resp)
    verdict = "retry"
    m = re.search(r"VERDICT:\s*(ok|retry)", critique, flags=re.IGNORECASE)
    if m:
        verdict = m.group(1).lower()

    out: dict[str, Any] = {
        "critique": critique,
        "verdict": verdict,
        "round": round_n,
        "status": [f"Critic round {round_n}: verdict={verdict}"],
    }
    # Reset research step budget when looping back for another team round
    if verdict == "retry":
        out["step"] = 0
    return out


def finalize_node(state: TeamState) -> dict[str, Any]:
    report = (state.get("draft") or "").strip()
    if not report:
        report = "(No report generated.)"
    report = _ensure_sources_footer(report, state.get("sources") or [])
    return {"report": report, "status": ["Team: final report ready"]}


def _ensure_sources_footer(report: str, sources: list[dict[str, Any]]) -> str:
    if "## Sources" in report or not sources:
        return report
    seen: set[str] = set()
    lines = ["", "## Sources"]
    for s in sources:
        url = (s.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (s.get("title") or url).strip()
        lines.append(f"- [{title}]({url})")
    if len(lines) <= 2:
        return report
    return report.rstrip() + "\n" + "\n".join(lines) + "\n"


def _route_after_researcher(state: TeamState) -> Literal["tools", "writer"]:
    messages = list(state.get("messages") or [])
    if not messages:
        return "writer"
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    step = int(state.get("step", 0))
    max_steps = int(state.get("max_steps", 8))
    if tool_calls and step < max_steps:
        return "tools"
    return "writer"


def _route_after_critic(state: TeamState) -> Literal["researcher", "finalize"]:
    verdict = (state.get("verdict") or "").lower()
    round_n = int(state.get("round", 0))
    max_rounds = int(state.get("max_rounds", 2))
    if verdict == "retry" and round_n < max_rounds:
        return "researcher"
    return "finalize"


def build_team_graph():
    g = StateGraph(TeamState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("tools", tools_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {"tools": "tools", "writer": "writer"},
    )
    g.add_edge("tools", "researcher")
    g.add_edge("writer", "critic")
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"researcher": "researcher", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


def run_team_research(
    topic: str,
    max_loops: int | None = None,
    max_rounds: int | None = None,
    on_status: StatusCallback | None = None,
) -> dict[str, Any]:
    """Run Planner → Researcher↔tools → Writer → Critic loop."""
    cfg = _cfg()
    if max_loops is None:
        max_steps = cfg["max_steps"]
    else:
        max_steps = max(4, int(max_loops) * 4) if int(max_loops) <= 3 else int(max_loops)
    if max_rounds is None:
        max_rounds = cfg["max_rounds"]

    graph = build_team_graph()
    state = initial_team_state(topic, max_steps=max_steps, max_rounds=int(max_rounds))
    if on_status:
        on_status(
            f"Team starting (max {max_steps} research steps, {max_rounds} critic rounds)"
        )

    final: dict[str, Any] = dict(state)
    for event in graph.stream(state, stream_mode="updates"):
        for _node, update in event.items():
            if not isinstance(update, dict):
                continue
            # Reset researcher step counter when critic sends work back
            if _node == "critic" and update.get("verdict") == "retry":
                final["step"] = 0
            for key in ("sources", "status"):
                if key in update and update[key] is not None:
                    final.setdefault(key, [])
                    chunk = update[key]
                    if isinstance(chunk, list):
                        final[key] = list(final.get(key) or []) + list(chunk)
                    else:
                        final[key] = list(final.get(key) or []) + [chunk]
            for key, val in update.items():
                if key in ("sources", "status"):
                    continue
                if key == "messages":
                    final.setdefault("messages", [])
                    if isinstance(val, list):
                        final["messages"] = list(final["messages"]) + list(val)
                    continue
                final[key] = val
            if on_status:
                for line in update.get("status") or []:
                    on_status(str(line))

    return final
