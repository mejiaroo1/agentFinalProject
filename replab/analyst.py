"""FeasibilityAnalyst: summarize paper/repo and emit a structured verdict."""

from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from replab.llm import llm
from replab.schemas import FeasibilityReport, PaperCandidate, Verdict
from replab.tools import github_repo_info

StatusCallback = Callable[[str], None]

_ANALYST_SYSTEM = """You are the FeasibilityAnalyst for a paper-with-code reproduction lab.
Given a paper abstract and GitHub repo inspection results, produce an honest
feasibility assessment for a student running on a typical laptop (CPU, ~8–16GB RAM,
limited disk). Prefer a short demo / CPU path when one exists.

CRITICAL: The entrypoint MUST use a .py file that appears in the provided py_files
list (or a documented module invocation). Never invent filenames.

Verdict rules:
- IMPOSSIBLE: no runnable entrypoint; stub/empty repo; private dataset only;
  multi-GPU / multi-day training with no reduced demo; archived empty repo.
- RISKY_BUT_POSSIBLE: large download, long but finite CPU job, outdated pins,
  or missing polish — warn clearly but do not refuse.
- READY: clear README, installable deps, verified entrypoint for a short run.

Always fill alarms with concrete warnings.
"""


def _py_basenames(py_files: list[str]) -> set[str]:
    names: set[str] = set()
    for path in py_files or []:
        names.add(path.replace("\\", "/").split("/")[-1].lower())
        names.add(path.replace("\\", "/").lower())
    return names


def _resolve_py_path(target: str, py_files: list[str]) -> str | None:
    """Map a README basename (e.g. classifier_sample.py) to repo-relative path."""
    target = (target or "").replace("\\", "/").lstrip("./")
    if not target:
        return None
    lower_map = {
        p.replace("\\", "/").lower(): p.replace("\\", "/") for p in (py_files or [])
    }
    if target.lower() in lower_map:
        return lower_map[target.lower()]
    base = target.split("/")[-1].lower()
    candidates = [
        p for p in lower_map.values() if p.split("/")[-1].lower() == base
    ]
    if not candidates:
        return None
    for pref in ("scripts/", "examples/", "example/", "demo/", "src/"):
        for c in candidates:
            if c.lower().startswith(pref):
                return c
    return candidates[0]


def _entrypoint_file(cmd: str) -> str | None:
    """Extract a .py path from a python/streamlit command, if present."""
    m = re.search(
        r"(?:python(?:3)?|streamlit\s+run)\s+([^\s]+\.py)",
        cmd or "",
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).replace("\\", "/").lstrip("./")


def _entrypoint_exists(cmd: str, py_files: list[str]) -> bool:
    target = _entrypoint_file(cmd)
    if not target:
        # e.g. python -m package or python -c ...
        if re.search(r"python(?:3)?\s+-m\s+\w+", cmd or "", re.I):
            return True
        if re.search(r"python(?:3)?\s+-c\s+", cmd or "", re.I):
            return True
        return False
    return _resolve_py_path(target, py_files) is not None


def _rewrite_entrypoint(cmd: str, py_files: list[str]) -> str:
    """Rewrite python script.pycalls to use the real repo-relative path."""
    target = _entrypoint_file(cmd)
    if not target:
        return (cmd or "").strip()
    resolved = _resolve_py_path(target, py_files)
    if not resolved or resolved == target:
        return (cmd or "").strip()
    return re.sub(
        re.escape(target),
        resolved,
        cmd,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


_PLACEHOLDER_TOKEN_RE = re.compile(
    r"^<[^>]+>$|^\$\{?\w+\}?$|^\[.*\]$|^(?:PATH|TODO|YOUR_.+)$",
    re.IGNORECASE,
)


def _is_placeholder_token(tok: str) -> bool:
    t = (tok or "").strip().strip("'\"")
    if not t:
        return True
    if _PLACEHOLDER_TOKEN_RE.match(t):
        return True
    if "<" in t or ">" in t:
        return True
    return False


def _sanitize_entrypoint_cmd(cmd: str) -> str:
    """Drop README placeholders like <lat_min> and $MODEL_FLAGS (unsafe in bash)."""
    cmd = (cmd or "").strip()
    if not cmd:
        return ""
    # Remove angle-bracket placeholders even if glued to text
    cmd = re.sub(r"<[^>\s]+>", " ", cmd)
    cmd = re.sub(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", " ", cmd)
    parts = []
    for tok in cmd.split():
        if _is_placeholder_token(tok):
            continue
        parts.append(tok)
    return " ".join(parts).strip()


def _example_numeric_args(readme: str, n: int = 4) -> list[str]:
    """Pull a concrete numeric demo bbox/args from README examples when present."""
    readme = readme or ""
    # load_scene(30.28, 30.29, -97.74, -97.73, ...)
    m = re.search(
        r"load_scene\(\s*"
        + r",\s*".join([r"([-+]?\d+(?:\.\d+)?)"] * n)
        + r"\s*[,)]",
        readme,
    )
    if m:
        return [m.group(i) for i in range(1, n + 1)]
    # python script.py 36.0 36.1 129.3 129.4
    m = re.search(
        r"python3?\s+[\w./-]+\.py\s+"
        + r"\s+".join([r"([-+]?\d+(?:\.\d+)?)"] * n),
        readme,
        flags=re.IGNORECASE,
    )
    if m:
        return [m.group(i) for i in range(1, n + 1)]
    # Markdown table row with 4 floats (skip header)
    for row in re.finditer(
        r"\|\s*[^|]+\s*\|\s*"
        + r"\|\s*".join([r"([-+]?\d+(?:\.\d+)?)"] * n)
        + r"\s*\|",
        readme,
    ):
        return [row.group(i) for i in range(1, n + 1)]
    return []


def _finalize_entrypoint(cmd: str, py_files: list[str], readme: str = "") -> str:
    """Sanitize placeholders, resolve paths, and fill demo numeric args if needed."""
    cmd = _sanitize_entrypoint_cmd(cmd)
    if not cmd:
        return ""
    cmd = _rewrite_entrypoint(cmd, py_files)
    # If command is only `python path.py` (no args) but README has a numeric demo,
    # attach those args — common for bbox CLIs documented with <lat_min> placeholders.
    parts = cmd.split()
    py_idx = next((i for i, t in enumerate(parts) if t.endswith(".py")), None)
    if py_idx is not None and len(parts) == py_idx + 1:
        nums = _example_numeric_args(readme, n=4)
        if nums:
            cmd = f"{cmd} {' '.join(nums)}"
    return cmd


def _pick_entrypoint(readme: str, py_files: list[str]) -> str:
    """Prefer README commands that point at real files; else common script names."""
    readme = readme or ""

    # Prefer documented Streamlit launches
    for m in re.finditer(
        r"streamlit\s+run\s+([\w./-]+\.py)",
        readme,
        flags=re.IGNORECASE,
    ):
        rel = m.group(1).replace("\\", "/").lstrip("./")
        resolved = _resolve_py_path(rel, py_files)
        if not resolved:
            # README often says `streamlit run app.py` from inside a subfolder
            for p in py_files or []:
                norm = p.replace("\\", "/")
                if norm.lower().endswith("/" + rel.lower()) or norm.lower() == rel.lower():
                    resolved = norm
                    break
        if resolved:
            return (
                f"streamlit run {resolved} --server.headless true "
                "--server.port 8501 --browser.gatherUsageStats false"
            )

    for pattern in (
        r"python3?\s+([\w./-]+\.py[^\n`]*)",
        r"`(python3?\s+[^`]+)`",
    ):
        for m in re.finditer(pattern, readme, flags=re.IGNORECASE):
            raw = m.group(1).strip()
            cmd = raw if raw.lower().startswith("python") else f"python {raw}"
            # Strip trailing markdown / shell line-continuation junk
            cmd = re.split(r"[`\n]", cmd)[0].strip()
            cmd = cmd.rstrip("\\").strip()
            cmd = _sanitize_entrypoint_cmd(cmd)
            if not cmd or not _entrypoint_exists(cmd, py_files):
                continue
            # Keep python + script + non-placeholder args (up to a few tokens)
            parts = cmd.split()
            cleaned = []
            for i, tok in enumerate(parts):
                cleaned.append(tok)
                if tok.endswith(".py") and i >= 1:
                    for extra in parts[i + 1 : i + 8]:
                        if _is_placeholder_token(extra):
                            break
                        cleaned.append(extra)
                    break
            finalized = _finalize_entrypoint(
                " ".join(cleaned).strip(), py_files, readme=readme
            )
            if finalized:
                return finalized

    preferred = (
        "demo.py",
        "example.py",
        "examples/demo.py",
        "main.py",
        "run.py",
        "train.py",
        "app.py",
        "scripts/image_sample.py",
        "scripts/classifier_sample.py",
    )
    lower_map = {p.replace("\\", "/").lower(): p for p in (py_files or [])}
    base_map: dict[str, str] = {}
    for p in py_files or []:
        norm = p.replace("\\", "/")
        base = norm.split("/")[-1].lower()
        # Prefer scripts/ when multiple same basenames
        if base not in base_map or norm.lower().startswith("scripts/"):
            base_map[base] = norm
    for name in preferred:
        path = None
        if name.lower() in lower_map:
            path = lower_map[name.lower()]
        elif name.lower() in base_map:
            path = base_map[name.lower()]
        if path:
            return _finalize_entrypoint(f"python {path}", py_files, readme=readme)

    # Prefer scripts/ over random nested files
    for p in py_files or []:
        norm = p.replace("\\", "/")
        if not norm.lower().startswith("scripts/"):
            continue
        if "train" in norm.lower():
            continue
        return _finalize_entrypoint(f"python {norm}", py_files, readme=readme)

    # Last resort: any non-test .py at repo root
    for p in py_files or []:
        norm = p.replace("\\", "/")
        if "/" in norm:
            continue
        if norm.startswith("test") or norm.endswith("_test.py"):
            continue
        if norm in ("setup.py", "conftest.py"):
            continue
        return _finalize_entrypoint(f"python {norm}", py_files, readme=readme)
    return ""


def _deterministic_signals(info: dict[str, Any]) -> dict[str, Any]:
    """Rule-based checks used for search-time screening and LLM biasing."""
    readme = (info.get("readme") or "").lower()
    req = (info.get("requirements_text") or "").lower()
    blob = f"{readme}\n{req}"
    files = info.get("files") or {}
    py_files = list(info.get("py_files") or [])
    size_kb = int(info.get("size_kb") or 0)
    alarms: list[str] = []
    reasons: list[str] = []

    has_req = bool(
        files.get("requirements.txt")
        or files.get("environment.yml")
        or files.get("pyproject.toml")
        or files.get("setup.py")
    )
    has_readme = bool(files.get("README") or info.get("readme"))

    if not has_readme:
        alarms.append("No README found — hard to discover how to run.")
    if not has_req:
        alarms.append("No requirements.txt / environment.yml / pyproject.toml detected.")
    if not py_files:
        alarms.append("No Python source files found in the repository tree.")
        reasons.append("Repository has no .py files to execute.")
    elif len(py_files) <= 1:
        # Often a stub (e.g. empty code.py only)
        alarms.append(f"Very few Python files ({len(py_files)}) — may be a stub repo.")

    if size_kb >= 500_000:
        alarms.append(f"Large repository size (~{size_kb // 1024} MB reported by GitHub).")
    elif size_kb >= 100_000:
        alarms.append(f"Moderate repository size (~{size_kb // 1024} MB).")

    cudaish = bool(
        re.search(r"\bcuda\b|\bcudnn\b|\bgpu\b|nvidia", blob)
        and not re.search(r"cpu[- ]?only|runs on cpu|no gpu required", blob)
    )
    if cudaish:
        alarms.append(
            "README/deps mention CUDA/GPU — may be incompatible on CPU-only machines."
        )

    if re.search(r"download.*(gb|dataset)|dataset.*(download|wget|curl)", blob):
        alarms.append("Possible large dataset download mentioned in docs.")

    if info.get("archived"):
        alarms.append("Repository is archived — may be unmaintained.")
        reasons.append("Archived GitHub repository.")

    entry = _pick_entrypoint(info.get("readme") or "", py_files)
    if entry and not _entrypoint_exists(entry, py_files):
        alarms.append(f"Discarded unverified entrypoint guess: {entry}")
        entry = ""

    if not entry:
        reasons.append("No verified runnable Python entrypoint in the repo.")

    return {
        "alarms": alarms,
        "reasons": reasons,
        "has_requirements": has_req,
        "has_readme": has_readme,
        "cuda_hint": cudaish,
        "entrypoint_guess": entry,
        "size_kb": size_kb,
        "py_files": py_files,
    }


def quick_feasibility(
    paper: PaperCandidate,
    info: dict[str, Any] | None = None,
) -> FeasibilityReport:
    """Fast rule-based feasibility (no LLM) for search-time screening."""
    if info is None:
        info = github_repo_info(paper.repo_url)
    if not info:
        return FeasibilityReport(
            summary=f"Could not inspect repo {paper.repo_url}.",
            alarms=["GitHub repository missing, private, or rate-limited."],
            verdict=Verdict.IMPOSSIBLE,
            reasons=["Repository could not be inspected."],
        )

    signals = _deterministic_signals(info)
    py_files = signals["py_files"]
    entry = signals["entrypoint_guess"]

    # Stub detection: tiny repo with no real scripts
    stub = (not py_files) or (len(py_files) <= 1 and not entry)

    if stub or not entry:
        verdict = Verdict.IMPOSSIBLE
    elif signals["cuda_hint"] or not signals["has_requirements"] or len(signals["alarms"]) >= 3:
        verdict = Verdict.RISKY_BUT_POSSIBLE
    else:
        verdict = Verdict.READY

    summary_bits = [
        info.get("description") or (paper.abstract[:240] if paper.abstract else paper.title),
        f"Verified entrypoint: `{entry}`" if entry else "No verified entrypoint.",
        f"{len(py_files)} Python file(s) in tree.",
    ]
    return FeasibilityReport(
        summary=" ".join(str(s) for s in summary_bits if s),
        cuda_required=bool(signals["cuda_hint"]),
        packages=[],
        entrypoint=entry,
        alarms=signals["alarms"],
        verdict=verdict,
        reasons=signals["reasons"]
        or (
            ["Looks runnable from README + file tree."]
            if verdict != Verdict.IMPOSSIBLE
            else ["Not suitable for automated reproduction."]
        ),
        has_requirements=signals["has_requirements"],
        has_readme=signals["has_readme"],
        repo_size_kb=signals["size_kb"],
        estimated_download_mb=round((signals["size_kb"] or 0) / 1024.0, 1)
        if signals["size_kb"]
        else None,
    )


def analyze_feasibility(
    paper: PaperCandidate,
    on_status: StatusCallback | None = None,
    use_llm: bool = True,
) -> FeasibilityReport:
    """Inspect the chosen paper's repo and return a FeasibilityReport."""

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    # Reuse search-time report when present and skipping LLM
    if paper.feasibility and not use_llm:
        return FeasibilityReport.model_validate(paper.feasibility)

    status(f"FeasibilityAnalyst: inspecting {paper.repo_url}...")
    info = github_repo_info(paper.repo_url)
    if not info:
        return FeasibilityReport(
            summary=f"Could not fetch public repo metadata for {paper.repo_url}.",
            alarms=["GitHub repository missing, private, or rate-limited."],
            verdict=Verdict.IMPOSSIBLE,
            reasons=["Repository could not be inspected."],
            has_requirements=False,
            has_readme=False,
        )

    # Always compute deterministic baseline first
    baseline = quick_feasibility(paper, info=info)
    signals = _deterministic_signals(info)

    if not use_llm:
        status(f"FeasibilityAnalyst: verdict={baseline.verdict.value} (rules only)")
        return baseline

    status("FeasibilityAnalyst: asking LLM for structured verdict...")
    context = (
        f"## Paper\nTitle: {paper.title}\nYear: {paper.year}\n"
        f"arXiv: {paper.arxiv_id}\nURL: {paper.paper_url}\n\n"
        f"Abstract:\n{paper.abstract[:2500]}\n\n"
        f"## Repo\n{info.get('full_name')} ({info.get('stars')} stars)\n"
        f"Language: {info.get('language')} | Size KB: {info.get('size_kb')} | "
        f"License: {info.get('license')}\n"
        f"Files present: {info.get('files')}\n"
        f"Python files (sample): {(info.get('py_files') or [])[:40]}\n"
        f"Deterministic baseline verdict: {baseline.verdict.value}\n"
        f"Deterministic entrypoint: {baseline.entrypoint}\n"
        f"Deterministic alarms: {baseline.alarms}\n\n"
        f"## README (truncated)\n{(info.get('readme') or '')[:6000]}\n\n"
        f"## Dependencies (truncated)\n{(info.get('requirements_text') or '')[:3000]}\n"
    )

    try:
        structured = llm().with_structured_output(FeasibilityReport)
        report: FeasibilityReport = structured.invoke(
            [
                SystemMessage(content=_ANALYST_SYSTEM),
                HumanMessage(content=context),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        status(f"FeasibilityAnalyst: structured output failed ({exc}); using rules only")
        return baseline

    # Merge + hard validation (never trust hallucinated entrypoints)
    merged_alarms = list(dict.fromkeys([*(report.alarms or []), *baseline.alarms]))
    report.alarms = merged_alarms
    report.has_requirements = baseline.has_requirements or report.has_requirements
    report.has_readme = baseline.has_readme or report.has_readme
    report.repo_size_kb = baseline.repo_size_kb

    py_files = list(info.get("py_files") or [])
    if report.entrypoint and not _entrypoint_exists(report.entrypoint, py_files):
        merged_alarms.append(
            f"Rejected hallucinated entrypoint `{report.entrypoint}` (file not in repo)."
        )
        report.alarms = list(dict.fromkeys(merged_alarms))
        report.entrypoint = baseline.entrypoint
    if not report.entrypoint:
        report.entrypoint = baseline.entrypoint

    # Never upgrade an IMPOSSIBLE baseline to READY without a verified entrypoint
    if baseline.verdict == Verdict.IMPOSSIBLE:
        report.verdict = Verdict.IMPOSSIBLE
        report.reasons = list(
            dict.fromkeys([*(report.reasons or []), *(baseline.reasons or [])])
        )
    elif report.verdict == Verdict.READY and not report.entrypoint:
        report.verdict = Verdict.IMPOSSIBLE
        report.reasons = list(
            dict.fromkeys([*(report.reasons or []), "No verified entrypoint after checks."])
        )

    if report.estimated_download_mb is None and baseline.estimated_download_mb is not None:
        report.estimated_download_mb = baseline.estimated_download_mb

    status(f"FeasibilityAnalyst: verdict={report.verdict.value}")
    return report
