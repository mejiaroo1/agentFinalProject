"""ExperimentRunner: Docker-sandboxed clone/install/run (local only)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from replab.entrypoint_intel import prepare_entrypoint_command
from replab.llm import llm, on_vercel
from replab.schemas import FeasibilityReport, PaperCandidate, RunResult

StatusCallback = Callable[[str], None]

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUNS_SUCCESSFUL = RUNS / "successful"
RUNS_FAILED = RUNS / "failed"
RUNS_IN_PROGRESS = RUNS / "in_progress"

_RETRY_SYSTEM = """You help recover a failed paper reproduction run inside Docker.
The repo is already cloned at /work. Propose ONE bash command (no markdown fences).

You will be given the entrypoint source when available — read it and honor its real CLI
(argparse positionals, flags, defaults). Fill required args with concrete demo values
from the docs/source. Never invent filenames or use placeholders like <lat_min> or other_script.py.
Prefer a short CPU smoke test when possible.
"""


def _ensure_run_roots() -> None:
    for path in (RUNS_SUCCESSFUL, RUNS_FAILED, RUNS_IN_PROGRESS):
        path.mkdir(parents=True, exist_ok=True)


def docker_available() -> bool:
    """True only if the Docker CLI can reach a running engine."""
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def docker_cli_present() -> bool:
    try:
        r = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def local_run_instructions(
    paper: PaperCandidate,
    report: FeasibilityReport,
) -> str:
    """Commands to show on Vercel (hybrid mode) or when Docker is missing."""
    entry = (report.entrypoint or "python demo.py").strip()
    slug = _slug(paper.title or paper.arxiv_id or "paper")
    return f"""## Run this locally (Vercel cannot clone/install/run repos)

Prerequisites: Docker Desktop, OpenAI key in `.env`, then:

```powershell
cd agentFinalProject
.\\.venv\\Scripts\\activate
# Prefer: open Gradio UI and use the Reproduction Lab tab
python app.py
```

Or from the Gradio **Reproduction Lab** tab on your machine:

1. Search / select: **{paper.title}**
2. Confirm feasibility (verdict was **{report.verdict.value}**)
3. Check permission and click **Run experiment**

Manual Docker sketch:

```bash
git clone {paper.repo_url} runs/in_progress/{slug}/repo
docker run --rm --memory 4g --cpus 2 -v "$PWD/runs/in_progress/{slug}/repo:/work" -w /work python:3.11-slim \\
  bash -c "pip install -r requirements.txt && {entry}"
```

Finished runs are filed under `runs/successful/` or `runs/failed/`.

Alarms:
""" + "\n".join(f"- {a}" for a in (report.alarms or ["(none)"]))


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return (s[:50] or "paper")


def _find_existing_run(slug: str) -> Path | None:
    """Locate a prior run folder (new layout or legacy flat runs/<slug>)."""
    _ensure_run_roots()
    category_roots = {
        RUNS_SUCCESSFUL.resolve(),
        RUNS_FAILED.resolve(),
        RUNS_IN_PROGRESS.resolve(),
        RUNS.resolve(),
    }
    for base in (RUNS_SUCCESSFUL, RUNS_FAILED, RUNS_IN_PROGRESS, RUNS):
        cand = base / slug
        if not cand.is_dir():
            continue
        if cand.resolve() in category_roots:
            continue
        if (cand / "repo").exists() or any(cand.iterdir()):
            return cand
    return None


def _run_dir(slug: str) -> Path:
    """Working directory under runs/in_progress/ (reuses prior clones when found)."""
    _ensure_run_roots()
    existing = _find_existing_run(slug)
    if existing is not None:
        parent = existing.parent.resolve()
        if parent in {RUNS_SUCCESSFUL.resolve(), RUNS_FAILED.resolve()}:
            dest = RUNS_IN_PROGRESS / slug
            if dest.exists() and dest.resolve() != existing.resolve():
                shutil.rmtree(dest, ignore_errors=True)
            if existing.resolve() != dest.resolve():
                shutil.move(str(existing), str(dest))
            path = dest
        else:
            path = existing
    else:
        path = RUNS_IN_PROGRESS / slug
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def _finalize_run_dir(run_dir: Path, success: bool) -> Path:
    """Move the run folder into runs/successful or runs/failed."""
    _ensure_run_roots()
    target_root = RUNS_SUCCESSFUL if success else RUNS_FAILED
    target = target_root / run_dir.name
    if run_dir.resolve() == target.resolve():
        return run_dir
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(run_dir), str(target))
    return target

def _clone_repo(repo_url: str, dest: Path, on_status: StatusCallback | None) -> None:
    if dest.exists() and any(dest.iterdir()):
        if on_status:
            on_status(f"ExperimentRunner: reusing existing clone at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if on_status:
        on_status(f"ExperimentRunner: cloning {repo_url}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _list_repo_py_files(repo_dir: Path, limit: int = 80) -> list[str]:
    out: list[str] = []
    if not repo_dir.exists():
        return out
    for p in sorted(repo_dir.rglob("*.py")):
        try:
            rel = p.relative_to(repo_dir).as_posix()
        except ValueError:
            continue
        if any(
            skip in f"/{rel}/"
            for skip in ("/venv/", "/.venv/", "/site-packages/", "/node_modules/")
        ):
            continue
        out.append(rel)
        if len(out) >= limit:
            break
    return out


def _resolve_script_in_repo(script: str, repo_dir: Path) -> str | None:
    script = (script or "").replace("\\", "/").lstrip("./")
    if not script:
        return None
    direct = repo_dir / script
    if direct.is_file():
        return script
    base = script.split("/")[-1]
    matches = [
        p.relative_to(repo_dir).as_posix()
        for p in repo_dir.rglob(base)
        if p.is_file()
    ]
    if not matches:
        return None
    for pref in ("scripts/", "examples/", "example/", "demo/", "src/"):
        for m in matches:
            if m.lower().startswith(pref):
                return m
    return matches[0]


def _rewrite_cmd_script_paths(cmd: str, repo_dir: Path) -> str:
    """Rewrite python foo.py -> python scripts/foo.py when needed."""
    m = re.search(r"(python(?:3)?\s+)([^\s]+\.py)", cmd or "", flags=re.IGNORECASE)
    if not m:
        return cmd
    resolved = _resolve_script_in_repo(m.group(2), repo_dir)
    if not resolved:
        return cmd
    return cmd[: m.start(2)] + resolved + cmd[m.end(2) :]


def _reject_placeholder_cmd(cmd: str) -> bool:
    low = (cmd or "").lower()
    if any(
        bad in low
        for bad in (
            "other_script.py",
            "<script",
            "<other",
            "<lat_",
            "<lon_",
            "your_script",
            "path/to/",
            "example_script.py",
        )
    ):
        return True
    # Angle-bracket placeholders are bash redirections / fake args
    if re.search(r"<[^>\s]+>", cmd or ""):
        return True
    return False


def _demo_args_from_repo(repo_dir: Path) -> list[str]:
    """Find a concrete lat/lon bbox demo from README / data docs."""
    from replab.analyst import _example_numeric_args

    blobs: list[str] = []
    for rel in ("README.md", "readme.md", "data/README.md", "docs/usage.md"):
        path = repo_dir / rel
        if path.is_file():
            try:
                blobs.append(path.read_text(encoding="utf-8", errors="replace")[:20000])
            except Exception:  # noqa: BLE001
                pass
    joined = "\n".join(blobs)
    return _example_numeric_args(joined, n=4)


def _docker_bash(repo_dir: Path, inner_cmd: str, timeout_sec: int) -> RunResult:
    """Run inner_cmd inside python:3.11-slim with the repo mounted at /work."""
    # Windows path for docker volume - Docker Desktop accepts //c/Users/... or native
    mount = str(repo_dir.resolve())
    cmd = [
        "docker",
        "run",
        "--rm",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "-v",
        f"{mount}:/work",
        "-w",
        "/work",
        "python:3.11-slim",
        "bash",
        "-c",
        inner_cmd,
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        duration = time.time() - started
        log = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        success = proc.returncode == 0
        # Streamlit headless smoke: server booted then timed out cleanly
        if (not success) and (
            "STREAMLIT_SMOKE_TIMEOUT_OK" in log
            or (
                "You can now view your Streamlit app" in log
                and "Traceback" not in log
                and "AttributeError" not in log
            )
        ):
            success = True
        return RunResult(
            command=" ".join(cmd[:10]) + f" ... bash -c {inner_cmd!r}",
            exit_code=proc.returncode,
            duration_sec=round(duration, 2),
            log_tail=log[-8000:],
            success=success,
            error="" if success else f"exit {proc.returncode}",
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started
        partial = ""
        if exc.stdout:
            partial += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(
                "utf-8", errors="replace"
            )
        if exc.stderr:
            partial += "\n" + (
                exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(
                    "utf-8", errors="replace"
                )
            )
        return RunResult(
            command=inner_cmd,
            exit_code=None,
            duration_sec=round(duration, 2),
            log_tail=(partial or "TIMEOUT")[-8000:],
            success=False,
            error=f"Timed out after {timeout_sec}s",
        )
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            command=inner_cmd,
            success=False,
            error=str(exc),
            log_tail=str(exc),
        )


def _deterministic_retry(failed: RunResult, entrypoint: str, repo_dir: Path) -> str | None:
    """Fix common paper-code / modern-deps mismatches without an LLM."""
    log = f"{failed.error or ''}\n{failed.log_tail or ''}"
    from replab.analyst import _sanitize_entrypoint_cmd

    entry = _sanitize_entrypoint_cmd(entrypoint or "")
    entry = _rewrite_cmd_script_paths(entry or "python run.py", repo_dir)

    # README placeholders leaked into the command (bash syntax error / fake args)
    if "syntax error near unexpected token" in log or re.search(r"<[^>\s]+>", entrypoint or ""):
        demo = _demo_args_from_repo(repo_dir)
        m = re.search(r"python(?:3)?\s+([^\s]+\.py)", entry, flags=re.I)
        if m and demo:
            script = _resolve_script_in_repo(m.group(1), repo_dir) or m.group(1)
            return (
                "export PYTHONPATH=/work:${PYTHONPATH:-}; "
                "pip install --no-cache-dir -r requirements.txt 2>/dev/null || "
                "pip install --no-cache-dir . 2>/dev/null || true; "
                f"python {script} {' '.join(demo)}"
            )
        if m:
            script = _resolve_script_in_repo(m.group(1), repo_dir) or m.group(1)
            return (
                "export PYTHONPATH=/work:${PYTHONPATH:-}; "
                f"python {script} --help"
            )

    # Streamlit run via bare python → missing ScriptRunContext / series_map None
    if (
        "missing ScriptRunContext" in log
        or "streamlit run" in log.lower()
        or ("No runtime found" in log and "streamlit" in log.lower())
    ):
        from replab.entrypoint_intel import prepare_entrypoint_command

        plan = prepare_entrypoint_command(repo_dir, entrypoint, use_llm=False)
        if plan.command and plan.command != entry:
            return plan.command

    # argparse: the following arguments are required: lon_max / ...
    if "arguments are required" in log.lower() or "the following arguments are required" in log:
        demo = _demo_args_from_repo(repo_dir)
        m = re.search(r"python(?:3)?\s+([^\s]+\.py)", entry, flags=re.I)
        if m and demo:
            script = _resolve_script_in_repo(m.group(1), repo_dir) or m.group(1)
            return (
                "export PYTHONPATH=/work:${PYTHONPATH:-}; "
                "pip install --no-cache-dir -r requirements.txt 2>/dev/null || "
                "pip install --no-cache-dir . 2>/dev/null || true; "
                f"python {script} {' '.join(demo)}"
            )

    # Wrong path: can't open file '/work/foo.py'
    m_missing = re.search(
        r"can't open file ['\"]?(?:/work/)?([^'\"]+\.py)['\"]?",
        log,
        flags=re.IGNORECASE,
    )
    if m_missing:
        resolved = _resolve_script_in_repo(m_missing.group(1), repo_dir)
        if resolved:
            demo = _demo_args_from_repo(repo_dir)
            args = f" {' '.join(demo)}" if demo else ""
            return (
                "export PYTHONPATH=/work:${PYTHONPATH:-}; "
                "pip install --no-cache-dir -r requirements.txt 2>/dev/null || "
                "pip install --no-cache-dir . 2>/dev/null || true; "
                f"python {resolved}{args}"
            )

    # gensim: LabeledSentence removed in gensim 4.x (renamed to TaggedDocument)
    if "LabeledSentence" in log and "gensim" in log.lower():
        for alt in ("word2vec-sentiments.py", "word2vec-sentiment.py", "main.py"):
            if (repo_dir / alt).exists():
                return (
                    "pip install --no-cache-dir gensim && "
                    f"python {alt}"
                )
        return f"pip install --no-cache-dir 'gensim<4.0.0' && {entry}"

    # Generic missing module: ModuleNotFoundError: No module named 'X'
    m = re.search(r"No module named ['\"]([A-Za-z0-9_]+)['\"]", log)
    if m:
        pkg = m.group(1)
        # Local package sitting in repo root — prefer PYTHONPATH over pip
        if (repo_dir / pkg).is_dir() or (repo_dir / f"{pkg}.py").is_file():
            return f"export PYTHONPATH=/work:${{PYTHONPATH:-}}; {entry}"
        # map a few import names to pip names
        pip_name = {"sklearn": "scikit-learn", "cv2": "opencv-python"}.get(pkg, pkg)
        return f"pip install --no-cache-dir {pip_name} && {entry}"

    return None


def _suggest_retry_command(
    failed: RunResult,
    entrypoint: str,
    repo_dir: Path | None = None,
) -> str | None:
    if repo_dir is not None:
        det = _deterministic_retry(failed, entrypoint, repo_dir)
        if det:
            return det
    try:
        py_hint = ""
        src_hint = ""
        py_files: list[str] = []
        if repo_dir and repo_dir.exists():
            py_files = _list_repo_py_files(repo_dir)
            if py_files:
                py_hint = "Python files in /work:\n- " + "\n- ".join(py_files[:40])
            m = re.search(r"python(?:3)?\s+([^\s]+\.py)", entrypoint or "", flags=re.I)
            if m:
                sp = repo_dir / (_resolve_script_in_repo(m.group(1), repo_dir) or m.group(1))
                if sp.is_file():
                    try:
                        src_hint = sp.read_text(encoding="utf-8", errors="replace")[:4000]
                    except Exception:  # noqa: BLE001
                        src_hint = ""
        resp = llm().invoke(
            [
                SystemMessage(content=_RETRY_SYSTEM),
                HumanMessage(
                    content=(
                        f"Original entrypoint: {entrypoint}\n"
                        f"{py_hint}\n\n"
                        f"Entrypoint source (truncated):\n{src_hint or '(unavailable)'}\n\n"
                        f"Error: {failed.error}\n"
                        f"Log tail:\n{failed.log_tail[-3500:]}"
                    )
                ),
            ]
        )
        text = str(resp.content or "").strip()
        text = re.sub(r"^```(?:bash|sh)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
        # Keep only the first line if the model rambling
        text = text.splitlines()[0].strip() if text else ""
        if _reject_placeholder_cmd(text):
            return None
        if text and ("python" in text.lower() or "pip" in text.lower()):
            if repo_dir is not None:
                text = _rewrite_cmd_script_paths(text, repo_dir)
                # If it still references a missing .py, reject
                m = re.search(r"python(?:3)?\s+([^\s]+\.py)", text, flags=re.I)
                if m and _resolve_script_in_repo(m.group(1), repo_dir) is None:
                    return None
            return text
    except Exception:  # noqa: BLE001
        return None
    return None


def _install_and_run_cmd(entrypoint: str) -> str:
    entry = entrypoint.strip()
    # Prefer requirements if present; otherwise try pip install .
    # Always put /work on PYTHONPATH so package dirs work even if setup.py is broken
    # (e.g. openai/guided-diffusion uses py_modules incorrectly).
    ensure_streamlit = ""
    if "streamlit run" in entry:
        ensure_streamlit = "pip install --no-cache-dir 'streamlit>=1.28' || true; "
        if "timeout " not in entry:
            # Headless smoke: if the server is still up at timeout, treat as boot OK.
            entry = (
                "set +e; "
                f"timeout 75s {entry}; "
                "code=$?; "
                "if [ \"$code\" -eq 124 ]; then "
                "echo STREAMLIT_SMOKE_TIMEOUT_OK; exit 0; "
                "fi; "
                "exit $code"
            )
    return (
        "set -e; "
        "export PYTHONPATH=/work:${PYTHONPATH:-}; "
        "if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; "
        "elif [ -f pyproject.toml ] || [ -f setup.py ]; then "
        "pip install --no-cache-dir . || true; "
        "fi; "
        f"{ensure_streamlit}"
        f"{entry}"
    )


def run_experiment(
    paper: PaperCandidate,
    report: FeasibilityReport,
    permission: bool = False,
    timeout_sec: int | None = None,
    on_status: StatusCallback | None = None,
    param_overrides: dict[str, str] | None = None,
) -> RunResult:
    """Clone + Docker-run a minimal experiment. Local only."""

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    if on_vercel():
        return RunResult(
            success=False,
            error="Runner disabled on Vercel (hybrid mode).",
            log_tail=local_run_instructions(paper, report),
        )

    if not permission:
        return RunResult(
            success=False,
            error="Permission not granted - check the approval box before running.",
        )

    if report.verdict.value == "IMPOSSIBLE":
        return RunResult(
            success=False,
            error="Feasibility verdict is IMPOSSIBLE - refusing to run.",
            log_tail="\n".join(report.reasons or report.alarms or []),
        )

    if not docker_available():
        hint = local_run_instructions(paper, report)
        if docker_cli_present():
            err = (
                "Docker CLI is installed but the engine is not running. "
                "Start Docker Desktop and retry."
            )
        else:
            err = "Docker is not available. Install Docker Desktop and retry."
        return RunResult(
            success=False,
            error=err,
            log_tail=hint,
        )

    if shutil.which("git") is None:
        return RunResult(success=False, error="git is not on PATH.")

    timeout_sec = int(timeout_sec or os.getenv("RUN_TIMEOUT_SEC", "600"))
    slug = _slug(paper.title or paper.arxiv_id or "paper")
    status(
        f"Reasoning: I'll stage this attempt under runs/in_progress/{slug}, "
        "then move it to runs/successful or runs/failed when finished."
    )
    run_dir = _run_dir(slug)
    repo_dir = run_dir / "repo"

    try:
        status("Reasoning: next I need the author's public code on disk (git clone or reuse).")
        _clone_repo(paper.repo_url, repo_dir, on_status)
    except Exception as exc:  # noqa: BLE001
        result = RunResult(success=False, error=f"git clone failed: {exc}")
        final_dir = _finalize_run_dir(run_dir, success=False)
        result.run_dir = str(final_dir)
        status(f"Reasoning: clone failed — filing this attempt under {final_dir}.")
        return result

    raw_entry = (report.entrypoint or "python -c \"print('no entrypoint')\"").strip()
    status(
        "Reasoning: before Docker, I'll read the entrypoint script and docs "
        "so we pass the arguments the program actually expects."
    )
    plan = prepare_entrypoint_command(repo_dir, raw_entry, use_llm=True)
    for note in plan.notes[:8]:
        status(f"Reasoning: {note}")
    if plan.spec and plan.spec.required_positionals:
        req = ", ".join(a.name for a in plan.spec.required_positionals)
        status(
            f"Reasoning: the script's CLI requires: {req}. "
            "I'll fill those from README examples when possible."
        )
    entry = plan.command
    status(f"Reasoning: planned command → `{entry}`")
    if param_overrides:
        # Append CLI-style overrides: --key value
        extras = []
        for k, v in param_overrides.items():
            flag = k if k.startswith("-") else f"--{k.lstrip('-')}"
            extras.append(f"{flag} {v}")
        entry = f"{entry} " + " ".join(extras)
        status(f"Reasoning: applying parameter overrides → `{entry}`")

    # Ensure image exists (pull can take a while first time)
    status("Reasoning: ensuring the python:3.11-slim Docker image is available…")
    subprocess.run(
        ["docker", "pull", "python:3.11-slim"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )

    inner = _install_and_run_cmd(entry)
    status(
        "Reasoning: launching Docker — install deps inside the container, "
        f"then run: {entry}"
    )
    result = _docker_bash(repo_dir, inner, timeout_sec)
    result.command = entry

    log_path = run_dir / "logs" / f"run_{int(time.time())}.txt"
    log_path.write_text(result.log_tail or result.error or "", encoding="utf-8")
    result.log_path = str(log_path)

    final: RunResult = result
    if result.success:
        status("Reasoning: first attempt succeeded.")
        result.metrics = _extract_metrics(result.log_tail)
        final = result
    else:
        status(
            "Reasoning: first attempt failed — I'll inspect the error and "
            "try one deterministic/LLM repair using the script source."
        )
        alt = _suggest_retry_command(result, entry, repo_dir=repo_dir)
        if not alt:
            status("Reasoning: no safe retry command found; marking this run as failed.")
            final = result
        else:
            status(f"Reasoning: retrying with → `{alt}`")
            if "pip install" in alt:
                retry_inner = f"set -e; {alt}"
            else:
                retry_inner = _install_and_run_cmd(alt)
            retry = _docker_bash(repo_dir, retry_inner, timeout_sec)
            retry.command = alt
            retry_log = run_dir / "logs" / f"retry_{int(time.time())}.txt"
            retry_log.write_text(retry.log_tail or retry.error or "", encoding="utf-8")
            retry.log_path = str(retry_log)
            if retry.success:
                retry.metrics = _extract_metrics(retry.log_tail)
                status("Reasoning: retry succeeded.")
            else:
                status("Reasoning: retry still failed.")
            final = retry

    # File under successful/ or failed/
    final_dir = _finalize_run_dir(run_dir, success=final.success)
    final.run_dir = str(final_dir)
    # Fix log paths if the folder moved
    if final.log_path:
        log_name = Path(final.log_path).name
        moved_log = final_dir / "logs" / log_name
        if moved_log.exists():
            final.log_path = str(moved_log)
    status(
        f"Reasoning: archived this attempt under "
        f"{'runs/successful' if final.success else 'runs/failed'}/{slug}."
    )

    if final.success:
        status(
            "Reasoning: writing a plain-language summary of the paper, "
            "the entrypoint, and what the arguments mean…"
        )
        try:
            from replab.narrate import narrate_successful_run

            final.narration = narrate_successful_run(paper, report, final, plan=plan)
            status("Reasoning: summary ready for you below.")
        except Exception as exc:  # noqa: BLE001
            status(f"Reasoning: could not write success narration ({exc}).")
            final.narration = ""

    return final


def _extract_metrics(log: str) -> dict:
    """Pull simple numeric metrics from logs (accuracy, loss, f1, etc.)."""
    metrics: dict[str, float] = {}
    patterns = [
        (r"accuracy[:\s=]+([0-9]*\.?[0-9]+)", "accuracy"),
        (r"acc[:\s=]+([0-9]*\.?[0-9]+)", "acc"),
        (r"loss[:\s=]+([0-9]*\.?[0-9]+)", "loss"),
        (r"f1[:\s=]+([0-9]*\.?[0-9]+)", "f1"),
        (r"bleu[:\s=]+([0-9]*\.?[0-9]+)", "bleu"),
    ]
    for pat, name in patterns:
        matches = re.findall(pat, log or "", flags=re.IGNORECASE)
        if matches:
            try:
                metrics[name] = float(matches[-1])
            except ValueError:
                pass
    return metrics
