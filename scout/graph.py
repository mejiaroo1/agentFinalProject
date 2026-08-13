"""ReAct research agent: LLM decides when to call web_search / fetch_page."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from replab.llm import get_api_key, require_api_key
from scout.prompts import AGENT_SYSTEM, REPORT_FALLBACK
from scout.state import AgentState, initial_state
from scout.tools import AGENT_TOOLS, sources_from_tool_content

load_dotenv()

StatusCallback = Callable[[str], None]


def _cfg() -> dict[str, Any]:
    return {
        "api_key": get_api_key(),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "max_steps": int(os.getenv("MAX_AGENT_STEPS", os.getenv("MAX_LOOPS", "8"))),
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


def _llm_with_tools():
    return _llm().bind_tools(AGENT_TOOLS)


def agent_node(state: AgentState) -> dict[str, Any]:
    """LLM reasons; may emit tool calls or a final report."""
    step = int(state.get("step", 0)) + 1
    messages = list(state.get("messages") or [])
    if not messages:
        topic = state.get("topic", "")
        messages = [
            SystemMessage(content=AGENT_SYSTEM),
            HumanMessage(
                content=(
                    f"Research topic:\n{topic}\n\n"
                    "Use your tools to investigate, then write the final cited report."
                )
            ),
        ]

    # Near the step budget, nudge the model to stop tooling and write the report
    max_steps = int(state.get("max_steps", 8))
    invoke_messages = list(messages)
    if step >= max_steps - 1:
        invoke_messages = messages + [
            HumanMessage(
                content=(
                    "You are near the step limit. Do not call more tools. "
                    "Write the final markdown report now using what you already gathered."
                )
            )
        ]

    resp = _llm_with_tools().invoke(invoke_messages)
    status: list[str] = [f"Step {step}: agent reasoning"]

    tool_calls = getattr(resp, "tool_calls", None) or []
    if tool_calls:
        names = []
        for tc in tool_calls:
            name = tc.get("name", "tool")
            args = tc.get("args") or {}
            preview = args.get("query") or args.get("url") or ""
            names.append(f"{name}({preview})" if preview else name)
        status.append("Agent requested tools: " + "; ".join(names))
    else:
        status.append("Agent finished tool use — drafting report")

    # First turn needs to seed messages; later turns only append the AI reply
    if not state.get("messages"):
        return {
            "step": step,
            "messages": messages + [resp],
            "status": status,
        }
    return {"step": step, "messages": [resp], "status": status}


def tools_node(state: AgentState) -> dict[str, Any]:
    """Execute whatever tools the agent requested; track sources for the UI."""
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


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Take the agent's final text as the report; fallback LLM call if empty."""
    report = ""
    for msg in reversed(list(state.get("messages") or [])):
        if isinstance(msg, AIMessage) and not (getattr(msg, "tool_calls", None) or []):
            content = msg.content
            if isinstance(content, str) and content.strip():
                report = content.strip()
                break
            if isinstance(content, list):
                parts = [
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                ]
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    report = joined
                    break

    if not report:
        # Force a report from the transcript
        transcript = list(state.get("messages") or [])
        resp = _llm().invoke(
            transcript
            + [HumanMessage(content=REPORT_FALLBACK.format(topic=state.get("topic", "")))]
        )
        report = str(resp.content).strip()

    # Append a compact sources section if the model omitted URLs we tracked
    report = _ensure_sources_footer(report, state.get("sources") or [])
    return {"report": report, "status": ["Final report ready"]}


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


def _route_after_agent(state: AgentState) -> Literal["tools", "finalize"]:
    messages = list(state.get("messages") or [])
    if not messages:
        return "finalize"
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    step = int(state.get("step", 0))
    max_steps = int(state.get("max_steps", 8))
    if tool_calls and step < max_steps:
        return "tools"
    return "finalize"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)

    return g.compile()


def run_research(
    topic: str,
    max_loops: int | None = None,
    on_status: StatusCallback | None = None,
) -> dict[str, Any]:
    """Run the ReAct research agent. `max_loops` maps to max agent steps."""
    if max_loops is None:
        max_steps = _cfg()["max_steps"]
    else:
        # UI slider historically meant research loops; treat as agent steps (min 4)
        max_steps = max(4, int(max_loops) * 4) if int(max_loops) <= 3 else int(max_loops)

    graph = build_graph()
    state = initial_state(topic, max_steps=max_steps)
    if on_status:
        on_status(f"ReAct agent starting (max {max_steps} steps)")

    final: dict[str, Any] = dict(state)
    for event in graph.stream(state, stream_mode="updates"):
        for _node, update in event.items():
            if not isinstance(update, dict):
                continue
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
                    # Keep a running message list for debugging; not required by UI
                    final.setdefault("messages", [])
                    if isinstance(val, list):
                        final["messages"] = list(final["messages"]) + list(val)
                    continue
                final[key] = val
            if on_status:
                for line in update.get("status") or []:
                    on_status(str(line))

    return final
