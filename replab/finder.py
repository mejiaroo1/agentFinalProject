"""PaperFinder: retrieve recent research papers that link official public code."""

from __future__ import annotations

import os
from typing import Callable

from replab.analyst import quick_feasibility
from replab.schemas import PaperCandidate, Verdict
from replab.tools import (
    arxiv_get_by_id,
    arxiv_search,
    github_repo_info,
    openalex_search,
    pop_search_notes,
    semantic_scholar_search,
)

StatusCallback = Callable[[str], None]

_VERDICT_RANK = {
    Verdict.READY.value: 0,
    Verdict.RISKY_BUT_POSSIBLE.value: 1,
    Verdict.IMPOSSIBLE.value: 2,
    "": 3,
}


def _min_year() -> int:
    return int(os.getenv("MIN_PAPER_YEAR", "2021"))


def _flush_api_notes(status: StatusCallback) -> None:
    for note in pop_search_notes():
        status(f"Reasoning: API note — {note}")


def _collect_research_papers(
    query: str,
    pool: int,
    min_year: int,
    on_status: StatusCallback | None,
) -> list[dict]:
    """Merge OpenAlex / arXiv / Semantic Scholar hits that link public GitHub code."""

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    status(
        f"Reasoning: searching OpenAlex for recent arXiv works with GitHub links "
        f"(min_year={min_year}, pool={pool})…"
    )
    oa_hits = openalex_search(
        query,
        max_results=pool,
        min_year=min_year,
        require_github=True,
        arxiv_only=True,
    )
    _flush_api_notes(status)
    status(
        f"Reasoning: OpenAlex returned {len(oa_hits)} paper(s) that mention GitHub in metadata."
    )

    arxiv_hits: list[dict] = []
    enriched: list[dict] = []

    # Skip rate-limit-prone APIs when OpenAlex already filled the pool.
    if len(oa_hits) >= max(3, pool // 2):
        status(
            "Reasoning: enough OpenAlex hits — skipping arXiv/Semantic Scholar "
            "to avoid rate limits and keep this fast."
        )
    else:
        status(
            f"Reasoning: OpenAlex was thin, so I'll also query arXiv "
            f"(min_year={min_year}, pool={pool})…"
        )
        arxiv_hits = arxiv_search(
            query,
            max_results=pool,
            require_github=True,
            recent_first=True,
            min_year=min_year,
        )
        _flush_api_notes(status)
        status(
            f"Reasoning: arXiv contributed {len(arxiv_hits)} paper(s) with author GitHub links."
        )

        status("Reasoning: checking Semantic Scholar for more recent research hits…")
        s2_hits = semantic_scholar_search(
            query, max_results=min(pool, 10), min_year=min_year
        )
        _flush_api_notes(status)
        status(f"Reasoning: Semantic Scholar returned {len(s2_hits)} hit(s).")

        # Enrich a few S2 papers that have an arXiv id (paced — arXiv rate-limits hard).
        enrich_budget = 5
        for hit in s2_hits:
            if hit.get("github_urls"):
                enriched.append(hit)
                continue
            aid = hit.get("arxiv_id") or ""
            if not aid or enrich_budget <= 0:
                continue
            ax = arxiv_get_by_id(aid)
            enrich_budget -= 1
            if not ax:
                continue
            urls = ax.get("github_urls") or []
            if not urls:
                continue
            hit = {
                **hit,
                "abstract": ax.get("abstract") or hit.get("abstract") or "",
                "github_urls": urls,
                "paper_url": ax.get("paper_url") or hit.get("paper_url") or "",
                "keywords": ax.get("keywords") or [],
                "source": "semantic_scholar+arxiv",
            }
            enriched.append(hit)

        _flush_api_notes(status)
        status(
            f"Reasoning: after enrichment, {len(enriched)} Semantic Scholar paper(s) "
            "have usable GitHub links."
        )

    # Prefer OpenAlex (usually works), then arXiv, then S2; dedupe by arxiv_id / title
    merged: list[dict] = []
    seen: set[str] = set()
    for paper in oa_hits + arxiv_hits + enriched:
        key = (paper.get("arxiv_id") or "").lower() or (paper.get("title") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(paper)

    if not merged:
        status(
            "Reasoning: all sources returned 0 papers with GitHub links. "
            "If you see rate-limit notes above, wait ~1 minute and retry, "
            "or set SEMANTIC_SCHOLAR_API_KEY / GITHUB_TOKEN in .env."
        )
    return merged


def find_papers_with_code(
    query: str,
    max_results: int = 8,
    on_status: StatusCallback | None = None,
    screen_feasibility: bool = True,
    only_runnable: bool = True,
    arxiv_pool: int = 20,
    min_year: int | None = None,
) -> list[PaperCandidate]:
    """Find *research papers* with official public code (OpenAlex / arXiv / S2).

    - Prefers **recent** papers (default ``MIN_PAPER_YEAR``, usually 2021+) to reduce
      broken legacy dependency APIs.
    - Only keeps papers where a GitHub repo URL appears in metadata/abstract,
      not loose web-search \"tutorials\".
    - OpenAlex is primary (stable free API); arXiv export + Semantic Scholar are
      secondary and often rate-limited without keys.
    """
    query = (query or "").strip()
    if not query:
        return []

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)

    year_floor = int(min_year) if min_year is not None else _min_year()
    pool = max(int(arxiv_pool), int(max_results))
    status(
        f"Reasoning: looking for recent research papers about “{query}” "
        f"(year ≥ {year_floor}) that link public GitHub code."
    )
    raw = _collect_research_papers(query, pool=pool, min_year=year_floor, on_status=on_status)
    status(
        f"Reasoning: screening {len(raw)} research paper(s) for runnable public code "
        "(verified entrypoints only)…"
    )

    candidates: list[PaperCandidate] = []
    seen_repos: set[str] = set()
    skipped_impossible = 0
    skipped_no_repo = 0

    for paper in raw:
        runnable_n = len(
            [c for c in candidates if c.verdict != Verdict.IMPOSSIBLE.value]
        )
        if only_runnable and runnable_n >= max_results:
            break
        if not only_runnable and len(candidates) >= max_results:
            break

        title = paper.get("title") or ""
        github_urls = list(paper.get("github_urls") or [])
        repo_url = github_urls[0] if github_urls else None

        # Intentionally NO web-search fallback — that pulled GitHub tutorials.
        if not repo_url:
            skipped_no_repo += 1
            continue

        info = github_repo_info(repo_url)
        if not info:
            status(f"Reasoning: repo not found / private: {repo_url}")
            continue

        key = (info.get("url") or repo_url).lower()
        if key in seen_repos:
            continue
        seen_repos.add(key)

        year = paper.get("year")
        cand = PaperCandidate(
            title=title,
            arxiv_id=paper.get("arxiv_id") or "",
            year=year,
            keywords=list(paper.get("keywords") or [])[:8],
            abstract=(paper.get("abstract") or "")[:2000],
            paper_url=paper.get("paper_url") or "",
            repo_url=info.get("url") or repo_url,
            stars=info.get("stars"),
            venue=(paper.get("venue") or ", ".join((paper.get("keywords") or [])[:2])),
        )

        if screen_feasibility:
            report = quick_feasibility(cand, info=info)
            # Prefer flagging very old deps even if year is recent
            if year and year < year_floor:
                report.alarms = list(
                    dict.fromkeys(
                        [*(report.alarms or []), f"Paper year {year} is below min {year_floor}."]
                    )
                )
            cand.verdict = report.verdict.value
            cand.entrypoint = report.entrypoint
            cand.alarms = list(report.alarms or [])[:5]
            cand.feasibility = report.model_dump(mode="json")

            if only_runnable and report.verdict == Verdict.IMPOSSIBLE:
                skipped_impossible += 1
                status(
                    f"Reasoning: skip IMPOSSIBLE '{title[:45]}...' "
                    f"({'; '.join(report.reasons[:1]) or 'no entrypoint'})"
                )
                continue

            src = paper.get("source") or "research"
            status(
                f"Reasoning: keep [{cand.verdict}] ({year}) '{title[:40]}...' "
                f"-> {info.get('full_name')} ({cand.entrypoint or 'no entry'}) [{src}]"
            )
        else:
            status(
                f"Reasoning: kept '{title[:50]}' -> {info.get('full_name')} "
                f"({info.get('stars')} stars)"
            )

        candidates.append(cand)

    # Prefer READY, then recent year, then stars
    candidates.sort(
        key=lambda c: (
            _VERDICT_RANK.get(c.verdict, 9),
            -(c.year or 0),
            -(c.stars or 0),
        )
    )

    status(
        f"Reasoning: showing {len(candidates)} research candidate(s)"
        + (f" (skipped {skipped_impossible} IMPOSSIBLE" if skipped_impossible else "")
        + (f", {skipped_no_repo} without code link" if skipped_no_repo else "")
        + (")" if skipped_impossible or skipped_no_repo else "")
    )
    return candidates[:max_results]
