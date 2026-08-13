"""Gradio UI: Paper-with-Code Reproduction Lab + legacy Research Scout."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Generator

from dotenv import load_dotenv

from envsafe import sanitize_env

# Must run before importing gradio: it parses GRADIO_SERVER_PORT at import time
# and a blank value (easy to create in a hosting dashboard) raises ValueError.
load_dotenv()
sanitize_env()

import gradio as gr  # noqa: E402

from scout.graph import run_research  # noqa: E402
from scout_team.graph import run_team_research  # noqa: E402
from replab.analyst import analyze_feasibility  # noqa: E402
from replab.finder import find_papers_with_code  # noqa: E402
from replab.llm import (  # noqa: E402
    get_api_key,
    looks_like_openai_key,
    on_vercel,
    openai_api_key,
    openai_model,
    redact_secrets,
)
from replab.runner import (  # noqa: E402
    docker_available,
    local_run_instructions,
    run_experiment,
)
from replab.schemas import (  # noqa: E402
    FeasibilityReport,
    PaperCandidate,
    RunResult,
    Verdict,
)
from replab.tweaker import (  # noqa: E402
    extract_parameters,
    format_comparison,
    run_with_overrides,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
ON_VERCEL = bool(os.getenv("VERCEL"))

if not ON_VERCEL:
    OUTPUTS.mkdir(exist_ok=True)
    (ROOT / "runs" / "successful").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "failed").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "in_progress").mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_STEPS = int(os.getenv("MAX_AGENT_STEPS", os.getenv("MAX_LOOPS", "8")))
DEFAULT_MAX_ROUNDS = int(os.getenv("MAX_TEAM_ROUNDS", "2"))
PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

MODE_SINGLE = "Single ReAct agent"
MODE_TEAM = "Multi-agent team (Planner → Researcher → Writer → Critic)"

# Docker, git and a writable workspace do not exist on Vercel's serverless
# runtime, so the reproduction stages are surfaced as disabled instead of failing.
HOSTED_BANNER = """
## Hosted demo — reproduction runs are disabled here

**PaperFinder** and **FeasibilityAnalyst** work on this deployed site.
**ExperimentRunner** and **ParamTweaker** need Docker, `git`, and a writable
workspace, which Vercel's serverless functions do not provide — those controls
are disabled below.

| Stage | This site (Vercel) | Your machine (localhost) |
|---|---|---|
| Find papers with code | Yes | Yes |
| Feasibility report | Yes | Yes |
| Clone / install / run in Docker | **No** | Yes |
| Parameter tweaks + re-run | **No** | Yes |

**To actually reproduce a paper, run this app locally:**

```powershell
git clone <this-repo> && cd agentFinalProject
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py     # then open http://127.0.0.1:7860
```

Start **Docker Desktop** first; the runner needs it.
"""

LOCAL_BANNER = """
**Local mode — full pipeline available.** Clone, install, Docker run, and
parameter tweaks all work here. On the deployed Vercel site these reproduction
stages are disabled by design (no Docker on serverless).
"""


def _outputs_dir() -> Path:
    if ON_VERCEL:
        path = Path("/tmp/outputs")
        path.mkdir(parents=True, exist_ok=True)
        return path
    return OUTPUTS


def _slug(topic: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", topic.strip().lower()).strip("-")
    return (s[:60] or "report")


def _key_status_line(api_key: str | None) -> str:
    """UI-safe note about which key source will be used (never echo the secret)."""
    pasted = (api_key or "").strip()
    if pasted and looks_like_openai_key(pasted):
        return "Reasoning: using the API key you pasted (kept in-memory for this request only)."
    if pasted:
        return "Reasoning: pasted key looks invalid — check it starts with sk-…"
    if get_api_key() and looks_like_openai_key(get_api_key()):
        return "Reasoning: using OPENAI_API_KEY from the server environment."
    return "Reasoning: no API key yet — paste one above or set OPENAI_API_KEY on the server."


def _stream_work(
    work: Callable[[Callable[[str], None]], Any],
    *,
    empty_outputs: tuple[Any, ...],
    finish: Callable[[Any, list[str]], tuple[Any, ...]],
    intro: str = "Reasoning: starting…",
    api_key: str | None = None,
) -> Generator[tuple[Any, ...], None, None]:
    """Run blocking agent work in a thread; stream reasoning lines to the UI."""
    lines: list[str] = [intro, _key_status_line(api_key)]
    q: Queue = Queue()
    holder: dict[str, Any] = {}

    def on_status(msg: str) -> None:
        text = msg if str(msg).startswith("Reasoning:") else f"Reasoning: {msg}"
        text = redact_secrets(text, api_key)
        lines.append(text)
        q.put(("tick", "\n".join(lines)))

    def worker() -> None:
        # ContextVars do not cross threads — set the key inside the worker.
        with openai_api_key(api_key):
            try:
                holder["value"] = work(on_status)
            except Exception as exc:  # noqa: BLE001
                holder["error"] = redact_secrets(str(exc), api_key)
            finally:
                q.put(("done", None))

    yield ("\n".join(lines), *empty_outputs)
    threading.Thread(target=worker, daemon=True).start()
    while True:
        try:
            kind, payload = q.get(timeout=0.25)
        except Empty:
            yield ("\n".join(lines), *empty_outputs)
            continue
        if kind == "tick":
            yield (payload, *empty_outputs)
        elif kind == "done":
            break

    if "error" in holder:
        lines.append(f"FAILED: {holder['error']}")
        yield ("\n".join(lines), *empty_outputs)
        return

    final = finish(holder.get("value"), lines)
    # Redact any accidental key leakage in status text of the final tuple
    if final and isinstance(final[0], str):
        final = (redact_secrets(final[0], api_key), *final[1:])
    yield final

# ---------------------------------------------------------------------------
# Legacy Research Scout
# ---------------------------------------------------------------------------


def research(
    topic: str,
    max_steps: int,
    mode: str,
    max_rounds: int,
    api_key: str | None = None,
):
    topic = (topic or "").strip()
    if not topic:
        yield "Enter a research topic.", "", None
        return

    max_steps = int(max_steps)
    max_rounds = int(max_rounds)
    use_team = mode == MODE_TEAM
    status_lines: list[str] = [
        f"Starting {'team' if use_team else 'ReAct'} research on: {topic}",
        f"Max research steps: {max_steps}",
        _key_status_line(api_key),
    ]
    if use_team:
        status_lines.append(f"Max critic rounds: {max_rounds}")

    def on_status(line: str) -> None:
        status_lines.append(redact_secrets(line, api_key))

    try:
        with openai_api_key(api_key):
            if use_team:
                result = run_team_research(
                    topic,
                    max_loops=max_steps,
                    max_rounds=max_rounds,
                    on_status=on_status,
                )
            else:
                result = run_research(topic, max_loops=max_steps, on_status=on_status)
    except Exception as exc:  # noqa: BLE001
        err = (
            f"**Error:** {redact_secrets(str(exc), api_key)}\n\n"
            "Paste your OpenAI API key in the secure field at the top, "
            "or set `OPENAI_API_KEY` in `.env` / Vercel env vars."
        )
        yield "\n".join(status_lines + [f"FAILED: {redact_secrets(str(exc), api_key)}"]), err, None
        return

    report = result.get("report") or "(No report generated.)"
    yield "\n".join(status_lines), redact_secrets(str(report), api_key), report


def save_report(report: str | None, topic: str) -> str:
    if not report or not str(report).strip():
        return "Nothing to save — run a research job first."
    topic = (topic or "report").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = _outputs_dir() / f"{stamp}-{_slug(topic)}.md"
    path.write_text(str(report).strip() + "\n", encoding="utf-8")
    return f"Saved to `{path}`"


# ---------------------------------------------------------------------------
# Reproduction Lab helpers
# ---------------------------------------------------------------------------


def _candidates_table(papers: list[PaperCandidate]) -> list[list[Any]]:
    rows = []
    for i, p in enumerate(papers):
        alarm = (p.alarms[0] if p.alarms else "")[:80]
        rows.append(
            [
                i,
                p.verdict or "?",
                p.title,
                p.year or "",
                p.entrypoint or "—",
                p.repo_url,
                p.stars if p.stars is not None else "",
                alarm,
            ]
        )
    return rows


def _feasibility_md(paper: PaperCandidate, report: FeasibilityReport) -> str:
    alarms = "\n".join(f"- {a}" for a in (report.alarms or ["(none)"]))
    reasons = "\n".join(f"- {r}" for r in (report.reasons or ["(none)"]))
    pkgs = ", ".join(report.packages) if report.packages else "(not listed)"
    return f"""### {paper.title}

**Verdict:** `{report.verdict.value}`

{report.summary}

| Field | Value |
|-------|-------|
| Repo | {paper.repo_url} |
| Stars | {paper.stars} |
| Python | {report.python_version} |
| CUDA required | {report.cuda_required} |
| Est. download (MB) | {report.estimated_download_mb} |
| Est. disk (MB) | {report.estimated_disk_mb} |
| Est. runtime (min) | {report.estimated_runtime_minutes} |
| Entrypoint | `{report.entrypoint or "—"}` |
| Has README | {report.has_readme} |
| Has requirements | {report.has_requirements} |
| Key packages | {pkgs} |

#### Alarms
{alarms}

#### Reasons
{reasons}
"""


def lab_search(query: str, api_key: str | None = None):
    query = (query or "").strip()
    if not query:
        yield "Enter a topic / keywords.", [], None, "Select a row after searching."
        return

    def work(on_status):
        return find_papers_with_code(
            query,
            max_results=8,
            arxiv_pool=20,
            screen_feasibility=True,
            only_runnable=True,
            on_status=on_status,
        )

    def finish(papers, lines):
        if not papers:
            lines.append(
                "Reasoning: no READY/RISKY papers with a verified entrypoint found. "
                "Try different keywords (classic ML demos work best)."
            )
            return "\n".join(lines), [], None, "No runnable candidates."

        state = [p.model_dump() for p in papers]
        ready_n = sum(1 for p in papers if p.verdict == Verdict.READY.value)
        risky_n = sum(1 for p in papers if p.verdict == Verdict.RISKY_BUT_POSSIBLE.value)
        hint = (
            f"Showing {len(papers)} screened candidate(s): "
            f"{ready_n} READY, {risky_n} RISKY. "
            "Click a row to load its feasibility report (already computed)."
        )
        lines.append(
            f"Reasoning: done searching — {ready_n} READY and {risky_n} RISKY "
            "candidates ready for you."
        )
        return "\n".join(lines), _candidates_table(papers), state, hint

    yield from _stream_work(
        work,
        empty_outputs=([], None, "Working…"),
        finish=finish,
        intro=(
            f'Reasoning: you asked about "{query}". '
            "I'll find recent papers with public code…"
        ),
        api_key=api_key,
    )


def lab_analyze(candidates_state, selection, api_key: str | None = None):
    if not candidates_state:
        yield "Search first.", None, None, True, ""
        return

    papers = [PaperCandidate.model_validate(p) for p in candidates_state]
    idx = 0
    if selection is not None:
        try:
            if isinstance(selection, (list, tuple)) and selection:
                idx = int(selection[0])
            else:
                idx = int(selection)
        except Exception:  # noqa: BLE001
            idx = 0
    idx = max(0, min(idx, len(papers) - 1))
    paper = papers[idx]

    def work(on_status):
        on_status(f"Selected [{idx}] {paper.title} (search verdict: {paper.verdict}).")
        if paper.feasibility:
            report = FeasibilityReport.model_validate(paper.feasibility)
            on_status("Using search-time feasibility (file-verified entrypoint).")
            return report
        on_status("No cached report — running a deeper feasibility pass…")
        return analyze_feasibility(paper, on_status=on_status, use_llm=True)

    def finish(report, lines):
        impossible = report.verdict == Verdict.IMPOSSIBLE
        md = _feasibility_md(paper, report)
        paper_state = {"paper": paper.model_dump(), "report": report.model_dump()}
        return "\n".join(lines), md, paper_state, impossible, ""

    yield from _stream_work(
        work,
        empty_outputs=(None, None, True, ""),
        finish=finish,
        intro="Reasoning: loading feasibility for the selected paper…",
        api_key=api_key,
    )


def lab_select_show(evt: gr.SelectData, candidates_state, api_key: str | None = None):
    if evt is None or not candidates_state:
        return 0, "No selection.", "Search first.", None, True

    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    idx = int(row)
    papers = [PaperCandidate.model_validate(p) for p in candidates_state]
    idx = max(0, min(idx, len(papers) - 1))
    paper = papers[idx]
    hint = f"Selected row {idx}: {paper.title}"

    with openai_api_key(api_key):
        if paper.feasibility:
            report = FeasibilityReport.model_validate(paper.feasibility)
        else:
            report = analyze_feasibility(paper, use_llm=False)

    md = _feasibility_md(paper, report)
    paper_state = {"paper": paper.model_dump(), "report": report.model_dump()}
    impossible = report.verdict == Verdict.IMPOSSIBLE
    return idx, hint, md, paper_state, impossible


def lab_run(paper_state, permission: bool, api_key: str | None = None):
    if not paper_state:
        yield "Select a screened paper first (click a table row).", "", None, None, ""
        return

    paper = PaperCandidate.model_validate(paper_state["paper"])
    report = FeasibilityReport.model_validate(paper_state["report"])

    if on_vercel() or ON_VERCEL:
        md = local_run_instructions(paper, report)
        yield (
            "Reasoning: hybrid mode — runner disabled on Vercel; showing local instructions.",
            md,
            paper_state,
            None,
            "",
        )
        return

    if report.verdict == Verdict.IMPOSSIBLE:
        yield (
            "Reasoning: refusing to run — feasibility verdict is IMPOSSIBLE.",
            _feasibility_md(paper, report),
            paper_state,
            None,
            "",
        )
        return

    if not permission:
        yield (
            "Reasoning: waiting on you — check **I approve clone/install/run** first.",
            "",
            paper_state,
            None,
            "",
        )
        return

    def work(on_status):
        result = run_experiment(paper, report, permission=True, on_status=on_status)
        params_json = "{}"
        try:
            on_status("Extracting tweakable parameters from the README/entrypoint…")
            plan = extract_parameters(paper, report, on_status=on_status)
            params_json = json.dumps(
                {p.name: p.default for p in plan.parameters},
                indent=2,
            )
            if plan.notes:
                on_status(f"ParamTweaker notes: {plan.notes}")
        except Exception as exc:  # noqa: BLE001
            on_status(f"ParamTweaker skipped: {exc}")
        return result, params_json

    def finish(payload, lines):
        result, params_json = payload
        narration_block = ""
        if result.success and result.narration:
            narration_block = (
                "### What this paper & code are about\n\n"
                f"{result.narration}\n\n"
            )
        log_md = f"""{narration_block}### Run result

- **Success:** {result.success}
- **Exit:** {result.exit_code}
- **Duration (s):** {result.duration_sec}
- **Error:** {result.error or "—"}
- **Run folder:** `{result.run_dir or "—"}`
- **Log file:** `{result.log_path or "—"}`
- **Metrics:** {result.metrics or "{}"}

```
{result.log_tail[-5000:] if result.log_tail else "(empty)"}
```
"""
        new_state = {**paper_state, "baseline": result.model_dump()}
        return (
            "\n".join(lines),
            redact_secrets(log_md, api_key),
            new_state,
            result.model_dump(),
            params_json,
        )

    yield from _stream_work(
        work,
        empty_outputs=("", paper_state, None, "{}"),
        finish=finish,
        intro=(
            f'Reasoning: starting a Docker reproduction for "{paper.title[:80]}"…'
        ),
        api_key=api_key,
    )


def lab_tweak(paper_state, params_json: str, permission: bool, api_key: str | None = None):
    if not paper_state or "paper" not in paper_state:
        yield "Run an experiment first (or analyze + approve).", "", None
        return

    paper = PaperCandidate.model_validate(paper_state["paper"])
    report = FeasibilityReport.model_validate(paper_state["report"])
    baseline = None
    if paper_state.get("baseline"):
        baseline = RunResult.model_validate(paper_state["baseline"])

    try:
        overrides = json.loads(params_json or "{}")
        if not isinstance(overrides, dict):
            raise ValueError("Parameters must be a JSON object of name → value")
        overrides = {str(k): str(v) for k, v in overrides.items()}
    except Exception as exc:  # noqa: BLE001
        yield f"Invalid parameters JSON: {exc}", "", paper_state
        return

    if on_vercel() or ON_VERCEL:
        yield "Tweaks require local Docker runner (hybrid mode).", "", paper_state
        return

    def work(on_status):
        return run_with_overrides(
            paper,
            report,
            overrides,
            permission=bool(permission),
            on_status=on_status,
            base_result=baseline,
        )

    def finish(payload, lines):
        result, comparison = payload
        new_state = {**paper_state, "tweaked": result.model_dump()}
        return "\n".join(lines), redact_secrets(str(comparison), api_key), new_state

    yield from _stream_work(
        work,
        empty_outputs=("", paper_state),
        finish=finish,
        intro="Reasoning: re-running with your parameter tweaks…",
        api_key=api_key,
    )


def build_ui() -> gr.Blocks:
    model = openai_model()
    hosted = ON_VERCEL or on_vercel()
    if hosted:
        env_note = "**Hosted on Vercel — reproduction runs are localhost-only.**"
    elif docker_available():
        env_note = "Running locally. Docker: detected."
    else:
        env_note = (
            "Running locally. Docker: not detected — "
            "start Docker Desktop before using the runner."
        )

    with gr.Blocks(title="Paper Reproduction Lab") as demo:
        gr.Markdown(
            f"""
# Paper-with-Code Reproduction Lab
**PaperFinder** screens feasibility while searching (verified entrypoints only).
Then: **you pick → permission → ExperimentRunner → ParamTweaker**.
Agent reasoning streams live into the status panels (no progress bar).
Runs are filed under `runs/successful/` or `runs/failed/`.
LLM: **OpenAI** (`{model}`). {env_note}
"""
        )

        with gr.Accordion("OpenAI API key (secure)", open=True):
            api_key = gr.Textbox(
                label="OpenAI API key",
                type="password",
                placeholder="sk-… (paste your key — never stored on disk by this app)",
                info=(
                    "Masked field. Used only in-memory for this browser session’s requests. "
                ),
                lines=1,
                max_lines=1,
            )

        with gr.Tab("Reproduction Lab"):
            gr.Markdown(HOSTED_BANNER if hosted else LOCAL_BANNER)
            with gr.Row():
                query = gr.Textbox(
                    label="Topic / keywords",
                    placeholder='e.g. "diffusion models" or "retrieval augmented generation"',
                    lines=2,
                    scale=4,
                )
                search_btn = gr.Button("Find runnable papers", variant="primary", scale=1)

            search_status = gr.Textbox(
                label="Agent reasoning (PaperFinder)",
                lines=14,
                interactive=False,
            )
            candidates = gr.Dataframe(
                headers=[
                    "#",
                    "Verdict",
                    "Title",
                    "Year",
                    "Entrypoint",
                    "Repo",
                    "Stars",
                    "Top alarm",
                ],
                datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                label=(
                    "Screened candidates (IMPOSSIBLE stubs hidden). "
                    "Click a row to load the report."
                ),
                interactive=False,
                wrap=True,
            )
            candidates_state = gr.State(value=None)
            selected_idx = gr.Number(value=0, label="Selected row index", precision=0)
            select_hint = gr.Textbox(label="Selection", interactive=False)

            analyze_btn = gr.Button("Load / refresh feasibility for selected row")
            analyze_status = gr.Textbox(
                label="Agent reasoning (Feasibility)", lines=6, interactive=False
            )
            feasibility_md = gr.Markdown(label="Feasibility report")
            paper_state = gr.State(value=None)
            impossible_flag = gr.Checkbox(
                label="IMPOSSIBLE (run disabled)", value=False, visible=False
            )

            gr.Markdown(
                "### ExperimentRunner — localhost only"
                + (
                    "\n\nDisabled on this hosted site. Clone the repo and run "
                    "`python app.py` on a machine with Docker to use this stage."
                    if hosted
                    else ""
                )
            )
            permission = gr.Checkbox(
                label=(
                    "Unavailable here — clone / install / run requires local Docker"
                    if hosted
                    else "I approve clone / install / run in Docker "
                    "(may download packages & use disk/CPU)"
                ),
                value=False,
                interactive=not hosted,
            )
            run_btn = gr.Button(
                "Run experiment (localhost only)" if hosted else "Run experiment",
                variant="secondary" if hosted else "primary",
                interactive=not hosted,
            )
            run_status = gr.Textbox(
                label="Agent reasoning (ExperimentRunner)", lines=14, interactive=False
            )
            run_md = gr.Markdown()
            baseline_state = gr.State(value=None)

            gr.Markdown(
                "### ParamTweaker — localhost only"
                + (
                    "\n\nRe-running with tweaks also needs the local Docker runner."
                    if hosted
                    else ""
                )
            )
            params_box = gr.Code(
                label="Parameters JSON (edit values, then re-run)",
                language="json",
                value="{}",
            )
            tweak_btn = gr.Button(
                "Re-run with tweaks (localhost only)" if hosted else "Re-run with tweaks",
                interactive=not hosted,
            )
            tweak_status = gr.Textbox(
                label="Agent reasoning (ParamTweaker)", lines=8, interactive=False
            )
            comparison_md = gr.Markdown()

            search_btn.click(
                fn=lab_search,
                inputs=[query, api_key],
                outputs=[search_status, candidates, candidates_state, select_hint],
            )
            candidates.select(
                fn=lab_select_show,
                inputs=[candidates_state, api_key],
                outputs=[
                    selected_idx,
                    select_hint,
                    feasibility_md,
                    paper_state,
                    impossible_flag,
                ],
            )
            analyze_btn.click(
                fn=lab_analyze,
                inputs=[candidates_state, selected_idx, api_key],
                outputs=[
                    analyze_status,
                    feasibility_md,
                    paper_state,
                    impossible_flag,
                    select_hint,
                ],
            )
            run_btn.click(
                fn=lab_run,
                inputs=[paper_state, permission, api_key],
                outputs=[run_status, run_md, paper_state, baseline_state, params_box],
            )
            tweak_btn.click(
                fn=lab_tweak,
                inputs=[paper_state, params_box, permission, api_key],
                outputs=[tweak_status, comparison_md, paper_state],
            )

        with gr.Tab("Research Scout"):
            gr.Markdown(
                """
Legacy scout: **Single ReAct** or **Team** (Planner → Researcher → Writer → Critic).
"""
            )
            mode = gr.Radio(
                choices=[MODE_SINGLE, MODE_TEAM],
                value=MODE_SINGLE,
                label="Execution mode",
            )
            with gr.Row():
                topic = gr.Textbox(
                    label="Research topic",
                    placeholder='e.g. "recent software updates in AI agents"',
                    lines=2,
                    scale=4,
                )
                max_steps = gr.Slider(
                    minimum=4,
                    maximum=12,
                    step=1,
                    value=min(max(DEFAULT_MAX_STEPS, 4), 12),
                    label="Max research steps",
                    scale=1,
                )
                max_rounds = gr.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=min(max(DEFAULT_MAX_ROUNDS, 1), 4),
                    label="Max critic rounds (team)",
                    scale=1,
                )
            run_research_btn = gr.Button("Run research", variant="primary")
            status = gr.Textbox(label="Agent reasoning", lines=12, interactive=False)
            report = gr.Markdown(label="Report")
            report_state = gr.State(value=None)
            with gr.Row():
                save_btn = gr.Button("Save report to outputs/")
                save_msg = gr.Textbox(label="Save status", interactive=False)

            run_research_btn.click(
                fn=research,
                inputs=[topic, max_steps, mode, max_rounds, api_key],
                outputs=[status, report, report_state],
            )
            save_btn.click(
                fn=save_report, inputs=[report_state, topic], outputs=[save_msg]
            )

    return demo


from fastapi import FastAPI

demo = build_ui()
_fastapi = FastAPI()
app = gr.mount_gradio_app(_fastapi, demo, path="/")


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT)
