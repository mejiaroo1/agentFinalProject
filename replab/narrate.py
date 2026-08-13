"""Plain-language narration after a successful reproduction run."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from replab.entrypoint_intel import EntrypointPlan, ScriptCliSpec
from replab.llm import llm
from replab.schemas import FeasibilityReport, PaperCandidate, RunResult

_NARRATE_SYSTEM = """You explain a successful paper-with-code reproduction to a student.
Write clear, simple English (no jargon walls). Use short paragraphs and light markdown.

Cover ALL of:
1) What the research paper is about (goal in plain terms — what problem, what idea).
2) What the code's overall goal is (what the program is trying to produce or demonstrate).
3) What the entrypoint command looks like and what each important argument means,
   tied back to that goal (why those args matter for the experiment).
4) One sentence on what this successful run achieved (from the log/metrics if useful).

Do not invent citations or results that are not supported by the provided materials.
"""


def _spec_blurb(spec: ScriptCliSpec | None) -> str:
    if not spec:
        return "(no CLI parse)"
    lines = []
    for a in spec.args:
        bit = f"- {a.kind} `{a.name}` required={a.required}"
        if a.default is not None:
            bit += f" default={a.default!r}"
        if a.choices:
            bit += f" choices={a.choices}"
        if a.help:
            bit += f" — {a.help[:120]}"
        lines.append(bit)
    return "\n".join(lines) or "(no argparse args found)"


def narrate_successful_run(
    paper: PaperCandidate,
    report: FeasibilityReport,
    result: RunResult,
    plan: EntrypointPlan | None = None,
) -> str:
    """LLM write-up: paper in simple terms + entrypoint/args vs code goal."""
    cmd = (plan.command if plan else "") or result.command or report.entrypoint
    filled = (plan.filled_args if plan else {}) or {}
    spec = plan.spec if plan else None
    try:
        resp = llm(temperature=0.3).invoke(
            [
                SystemMessage(content=_NARRATE_SYSTEM),
                HumanMessage(
                    content=(
                        f"Title: {paper.title}\n"
                        f"Year: {paper.year}\n"
                        f"arXiv: {paper.arxiv_id}\n"
                        f"Repo: {paper.repo_url}\n\n"
                        f"Abstract:\n{(paper.abstract or '')[:1800]}\n\n"
                        f"Feasibility summary:\n{report.summary}\n\n"
                        f"Entrypoint command used:\n{cmd}\n"
                        f"Filled args: {filled}\n"
                        f"CLI inspection:\n{_spec_blurb(spec)}\n\n"
                        f"Run success={result.success} duration={result.duration_sec}s "
                        f"metrics={result.metrics}\n"
                        f"Log tail (truncated):\n{(result.log_tail or '')[-2500:]}\n"
                    )
                ),
            ]
        )
        text = str(resp.content or "").strip()
        return text or "_Could not generate a narration._"
    except Exception as exc:  # noqa: BLE001
        # Deterministic fallback if LLM unavailable
        req = []
        if spec:
            req = [a.name for a in spec.required_positionals]
        return (
            f"### What this paper is about\n\n"
            f"{(paper.abstract or report.summary or paper.title)[:600]}\n\n"
            f"### Entrypoint\n\n"
            f"We ran:\n```\n{cmd}\n```\n\n"
            f"Required arguments detected: {', '.join(req) or 'none / optional flags only'}.\n"
            f"Filled values: {filled or '—'}.\n\n"
            f"_(Narration LLM unavailable: {exc})_"
        )
