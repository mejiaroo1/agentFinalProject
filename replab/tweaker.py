"""ParamTweaker: extract editable params, re-run with overrides, compare."""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from replab.llm import llm
from replab.runner import run_experiment
from replab.schemas import FeasibilityReport, PaperCandidate, RunResult, TweakPlan
from replab.tools import github_repo_info

StatusCallback = Callable[[str], None]

_TWEAK_SYSTEM = """You extract tweakable hyperparameters for a paper reproduction demo.
Prefer a small set (3–6) of CLI-friendly parameters such as epochs, batch_size,
learning_rate, max_steps, seed. Use names that can be passed as --name value.
If the README shows flags, reuse them. Keep defaults conservative for a short CPU run.
"""


def extract_parameters(
    paper: PaperCandidate,
    report: FeasibilityReport,
    on_status: StatusCallback | None = None,
) -> TweakPlan:
    """Ask the LLM for a ParamTweaker plan from README + entrypoint."""

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    status("ParamTweaker: extracting parameters...")
    info = github_repo_info(paper.repo_url) or {}
    readme = (info.get("readme") or "")[:6000]
    context = (
        f"Paper: {paper.title}\nRepo: {paper.repo_url}\n"
        f"Suggested entrypoint: {report.entrypoint}\n\n"
        f"README:\n{readme}"
    )
    try:
        structured = llm().with_structured_output(TweakPlan)
        plan: TweakPlan = structured.invoke(
            [
                SystemMessage(content=_TWEAK_SYSTEM),
                HumanMessage(content=context),
            ]
        )
    except Exception:  # noqa: BLE001
        plan = TweakPlan(
            entrypoint=report.entrypoint or "python train.py",
            parameters=[],
            notes="Could not extract structured parameters; edit the entrypoint manually.",
        )

    if not plan.entrypoint:
        plan.entrypoint = report.entrypoint
    status(f"ParamTweaker: {len(plan.parameters)} parameter(s)")
    return plan


def run_with_overrides(
    paper: PaperCandidate,
    report: FeasibilityReport,
    overrides: dict[str, str],
    permission: bool = True,
    on_status: StatusCallback | None = None,
    base_result: RunResult | None = None,
) -> tuple[RunResult, str]:
    """Re-run with parameter overrides; return (result, comparison markdown)."""
    # Temporarily override entrypoint if plan stored different one
    tweaked_report = report.model_copy(deep=True)
    result = run_experiment(
        paper,
        tweaked_report,
        permission=permission,
        on_status=on_status,
        param_overrides=overrides or None,
    )
    comparison = format_comparison(base_result, result, overrides)
    return result, comparison


def format_comparison(
    baseline: RunResult | None,
    tweaked: RunResult,
    overrides: dict[str, str] | None = None,
) -> str:
    lines = ["## Parameter tweak comparison", ""]
    if overrides:
        lines.append("**Overrides:** " + ", ".join(f"`{k}={v}`" for k, v in overrides.items()))
        lines.append("")
    lines.append("| Run | Success | Exit | Duration (s) | Metrics |")
    lines.append("|-----|---------|------|--------------|---------|")

    def row(label: str, r: RunResult | None) -> str:
        if not r:
            return f"| {label} | - | - | - | - |"
        metrics = ", ".join(f"{k}={v}" for k, v in (r.metrics or {}).items()) or "-"
        return (
            f"| {label} | {r.success} | {r.exit_code} | "
            f"{r.duration_sec} | {metrics} |"
        )

    lines.append(row("Baseline", baseline))
    lines.append(row("Tweaked", tweaked))
    lines.append("")
    if tweaked.error:
        lines.append(f"**Tweaked error:** {tweaked.error}")
    if tweaked.log_tail:
        lines.append("")
        lines.append("### Tweaked log (tail)")
        lines.append("```")
        lines.append(tweaked.log_tail[-3000:])
        lines.append("```")
    return "\n".join(lines)
