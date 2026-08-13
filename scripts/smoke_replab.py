"""Smoke tests for replab (finder, analyst, docker runner)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replab.analyst import analyze_feasibility
from replab.finder import find_papers_with_code
from replab.runner import docker_available, run_experiment
from replab.schemas import FeasibilityReport, PaperCandidate, Verdict


def main() -> int:
    print("docker", docker_available())
    papers = find_papers_with_code("word2vec", max_results=6, on_status=print)
    print("n_papers", len(papers))
    if not papers:
        print("FAIL: no papers with code")
        return 1
    for p in papers[:3]:
        print(" -", p.title[:70], "->", p.repo_url)
    report = analyze_feasibility(papers[0], on_status=print)
    print("verdict", report.verdict)
    print("entrypoint", (report.entrypoint or "")[:120])
    print("alarms_n", len(report.alarms))

    tiny = PaperCandidate(title="requests-smoke", repo_url="https://github.com/psf/requests")
    tiny_report = FeasibilityReport(
        summary="smoke",
        verdict=Verdict.READY,
        entrypoint='python -c "import sys; print(sys.version)"',
        has_readme=True,
        has_requirements=True,
    )
    result = run_experiment(
        tiny, tiny_report, permission=True, timeout_sec=420, on_status=print
    )
    print("run_success", result.success, "exit", result.exit_code, "err", result.error)
    print("log_snip", (result.log_tail or "")[-500:])
    if result.success:
        return 0
    # Soft-pass if Docker engine is down (common on CI / Desktop not started)
    err = (result.error or "") + (result.log_tail or "")
    if "engine is not running" in err or "dockerDesktopLinuxEngine" in err:
        print("SOFT PASS: Docker engine not running; finder/analyst OK")
        return 0
    if not docker_available():
        print("SOFT PASS: Docker unavailable; finder/analyst OK")
        return 0
    print("WARN: docker smoke failed with engine available")
    return 2


if __name__ == "__main__":
    os.environ.pop("VERCEL", None)
    sys.exit(main())
