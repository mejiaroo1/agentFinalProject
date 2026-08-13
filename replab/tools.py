"""Free APIs for paper search and GitHub repo inspection."""

from __future__ import annotations

import os
import random
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from scout.tools import web_search_raw

_GITHUB_RE = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")

# OpenAlex source id for arXiv (stable across their catalog)
_OPENALEX_ARXIV_SOURCE = "S4306400194"
_UA = "PaperReproductionLab/1.0 (student project; mailto:replab@localhost)"

# Minimum seconds between requests to the same host (polite pacing).
_HOST_MIN_INTERVAL = {
    "export.arxiv.org": 3.2,
    "api.semanticscholar.org": 1.1,
    "api.openalex.org": 0.25,
    "api.github.com": 0.4,
}

# Last errors surfaced to PaperFinder status (cleared by callers).
_last_search_notes: list[str] = []
_last_host_request_at: dict[str, float] = {}


def pop_search_notes() -> list[str]:
    """Return and clear API notes (rate limits, outages) from the last search calls."""
    notes = list(_last_search_notes)
    _last_search_notes.clear()
    return notes


def _note(msg: str) -> None:
    if msg and msg not in _last_search_notes:
        _last_search_notes.append(msg)


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Parse Retry-After as seconds, if present."""
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    ra = ra.strip()
    if ra.isdigit():
        return float(ra)
    return None


def _backoff_seconds(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = 60.0,
    retry_after: float | None = None,
) -> float:
    """Exponential backoff with equal jitter: min(cap, base*2^attempt), then jitter.

    Caps growth so we wait longer on each failure without slamming the API.
    Honors Retry-After when it asks for a longer wait.
    """
    exp = min(cap, base * (2**attempt))
    # Equal jitter: spreads concurrent clients, still ~exponential on average
    delay = (exp / 2.0) + random.uniform(0.0, exp / 2.0)
    if retry_after is not None:
        delay = max(delay, float(retry_after))
    return delay


def _pace_host(host: str) -> None:
    """Sleep if needed so we do not exceed per-host request cadence."""
    min_gap = float(_HOST_MIN_INTERVAL.get(host, 0.0))
    if min_gap <= 0:
        return
    last = _last_host_request_at.get(host, 0.0)
    gap = time.monotonic() - last
    if gap < min_gap:
        time.sleep(min_gap - gap)


def extract_github_urls(text: str) -> list[str]:
    """Return normalized github.com/owner/repo URLs found in text."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _GITHUB_RE.finditer(text or ""):
        owner, repo = m.group(1), m.group(2)
        repo = repo.rstrip(".,);]")
        if repo.endswith(".git"):
            repo = repo[:-4]
        # Skip non-repo paths
        if owner.lower() in {"topics", "search", "orgs", "settings", "features"}:
            continue
        url = f"https://github.com/{owner}/{repo}"
        key = url.lower()
        if key not in seen:
            seen.add(key)
            out.append(url)
    return out


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": _UA,
    }
    token = (os.getenv("GITHUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
    retries: int = 4,
    pace_arxiv: bool = False,
    backoff_base: float = 1.0,
    backoff_cap: float = 60.0,
) -> httpx.Response | None:
    """GET with host pacing + exponential backoff on 429/503/transient errors."""
    hdrs = {"User-Agent": _UA, "Accept": "*/*"}
    if headers:
        hdrs.update(headers)

    host = urlparse(url).netloc.lower()
    # Legacy flag: enforce arXiv's ~3s courtesy gap.
    if pace_arxiv:
        _HOST_MIN_INTERVAL.setdefault(host, 3.2)

    last_err = ""
    for attempt in range(max(1, int(retries))):
        _pace_host(host)
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, params=params, headers=hdrs)
            _last_host_request_at[host] = time.monotonic()

            if resp.status_code in (429, 503, 502, 504):
                last_err = f"HTTP {resp.status_code} from {host}"
                if attempt + 1 >= retries:
                    break
                wait = _backoff_seconds(
                    attempt,
                    base=backoff_base,
                    cap=backoff_cap,
                    retry_after=_retry_after_seconds(resp),
                )
                _note(f"{last_err}; exponential backoff {wait:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                last_err = f"HTTP {resp.status_code} from {host}"
                _note(last_err)
                return None
            return resp
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            _last_host_request_at[host] = time.monotonic()
            if attempt + 1 >= retries:
                break
            wait = _backoff_seconds(attempt, base=backoff_base, cap=backoff_cap)
            _note(f"{last_err}; exponential backoff {wait:.1f}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)

    if last_err:
        _note(last_err)
    return None


def _client_get_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    retries: int = 4,
    backoff_base: float = 1.0,
    backoff_cap: float = 45.0,
) -> httpx.Response | None:
    """GET on an existing client with host pacing + exponential backoff."""
    host = urlparse(url).netloc.lower()
    last_err = ""
    for attempt in range(max(1, int(retries))):
        _pace_host(host)
        try:
            resp = client.get(url)
            _last_host_request_at[host] = time.monotonic()
            if resp.status_code in (429, 503, 502, 504):
                last_err = f"HTTP {resp.status_code} from {host}"
                if attempt + 1 >= retries:
                    break
                wait = _backoff_seconds(
                    attempt,
                    base=backoff_base,
                    cap=backoff_cap,
                    retry_after=_retry_after_seconds(resp),
                )
                time.sleep(wait)
                continue
            return resp
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            _last_host_request_at[host] = time.monotonic()
            if attempt + 1 >= retries:
                break
            time.sleep(_backoff_seconds(attempt, base=backoff_base, cap=backoff_cap))
    if last_err:
        _note(last_err)
    return None


def arxiv_search(
    query: str,
    max_results: int = 8,
    *,
    require_github: bool = True,
    recent_first: bool = True,
    min_year: int | None = None,
) -> list[dict[str, Any]]:
    """Search arXiv Atom API (no key). Returns research-paper dicts.

    When ``require_github`` is True, prefer papers whose metadata mentions GitHub,
    then keep only those with an extractable repo URL.
    """
    query = (query or "").strip()
    if not query:
        return []

    # Build arXiv boolean query from keywords
    tokens = [t for t in re.split(r"\s+", query) if t]
    if len(tokens) == 1:
        topic = f"all:{tokens[0]}"
    else:
        topic = " AND ".join(f"all:{t}" for t in tokens[:8])
    if require_github:
        search_q = f"({topic}) AND all:github"
    else:
        search_q = topic

    sort = "submittedDate" if recent_first else "relevance"
    # Fetch a wider pool when filtering for GitHub URLs locally
    fetch_n = int(max_results) if not require_github else min(int(max_results) * 2, 40)
    params = {
        "search_query": search_q,
        "start": 0,
        "max_results": fetch_n,
        "sortBy": sort,
        "sortOrder": "descending",
    }
    resp = _http_get(
        "https://export.arxiv.org/api/query",
        params=params,
        timeout=45.0,
        retries=3,
        pace_arxiv=True,
    )
    if resp is None:
        _note("arXiv search failed (rate limit or outage). Trying other sources.")
        return []

    # arXiv sometimes returns 200 with a short HTML/text error body
    body = resp.text or ""
    if "Rate exceeded" in body or body.strip().startswith("<!DOCTYPE"):
        _note("arXiv rate-limited or unavailable.")
        return []

    papers = _parse_arxiv_atom(body)
    if min_year is not None:
        papers = [p for p in papers if (p.get("year") or 0) >= int(min_year)]
    if require_github:
        papers = [p for p in papers if p.get("github_urls")]
    for p in papers:
        p["source"] = "arxiv"
    return papers[: int(max_results)]


def arxiv_get_by_id(arxiv_id: str) -> dict[str, Any] | None:
    """Fetch a single arXiv paper by id (for enriching Semantic Scholar hits)."""
    arxiv_id = (arxiv_id or "").strip()
    m = _ARXIV_ID_RE.search(arxiv_id)
    if not m:
        return None
    aid = m.group(1)
    resp = _http_get(
        "https://export.arxiv.org/api/query",
        params={"id_list": aid, "start": 0, "max_results": 1},
        timeout=30.0,
        retries=2,
        pace_arxiv=True,
    )
    if resp is None:
        return None
    papers = _parse_arxiv_atom(resp.text)
    if not papers:
        return None
    papers[0]["source"] = "arxiv"
    return papers[0]


def _openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[int(i)] = word
    return " ".join(positions[i] for i in sorted(positions))


def openalex_search(
    query: str,
    max_results: int = 20,
    *,
    min_year: int | None = 2021,
    require_github: bool = True,
    arxiv_only: bool = True,
) -> list[dict[str, Any]]:
    """Search OpenAlex (generous free tier) for recent works with code links.

    Prefer arXiv-hosted works; reconstruct abstracts and extract GitHub URLs.
    Docs: https://docs.openalex.org/
    """
    query = (query or "").strip()
    if not query:
        return []

    year = int(min_year) if min_year is not None else 2021
    # Bias toward papers that mention code hosting in the search text
    search = query if "github" in query.lower() else f"{query} github"
    filters = [
        f"from_publication_date:{year}-01-01",
        "has_abstract:true",
    ]
    if arxiv_only:
        filters.append(f"locations.source.id:{_OPENALEX_ARXIV_SOURCE}")

    fetch_n = min(max(int(max_results) * 3, int(max_results)), 50)
    resp = _http_get(
        "https://api.openalex.org/works",
        params={
            "search": search,
            "filter": ",".join(filters),
            "per-page": fetch_n,
            "sort": "publication_date:desc",
            "mailto": "replab@localhost",
        },
        headers={"Accept": "application/json"},
        timeout=40.0,
        retries=3,
    )
    if resp is None:
        _note("OpenAlex search failed.")
        return []

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        _note("OpenAlex returned non-JSON.")
        return []

    papers: list[dict[str, Any]] = []
    for work in data.get("results") or []:
        title = re.sub(r"<[^>]+>", "", (work.get("display_name") or "")).strip()
        if not title:
            continue
        abstract = _openalex_abstract(work.get("abstract_inverted_index"))
        ids = work.get("ids") or {}
        arxiv_id = ""
        raw_arxiv = (ids.get("arxiv") or "").strip()
        if raw_arxiv:
            m = _ARXIV_ID_RE.search(raw_arxiv)
            if m:
                arxiv_id = m.group(1)
        if not arxiv_id:
            for loc in work.get("locations") or []:
                lp = loc.get("landing_page_url") or ""
                m = _ARXIV_ID_RE.search(lp)
                if m:
                    arxiv_id = m.group(1)
                    break
        blob = f"{title}\n{abstract}"
        github_urls = extract_github_urls(blob)
        if require_github and not github_urls:
            continue
        year_v = work.get("publication_year")
        paper_url = ""
        if arxiv_id:
            paper_url = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            paper_url = (ids.get("openalex") or work.get("id") or "") or ""
        papers.append(
            {
                "title": title,
                "arxiv_id": arxiv_id,
                "year": int(year_v) if year_v else None,
                "abstract": abstract[:2000],
                "paper_url": paper_url,
                "keywords": [],
                "comment": "",
                "github_urls": github_urls,
                "venue": "arXiv" if arxiv_id else (work.get("type") or ""),
                "source": "openalex",
            }
        )
        if len(papers) >= int(max_results):
            break
    return papers


def semantic_scholar_search(
    query: str,
    max_results: int = 10,
    *,
    min_year: int | None = 2021,
) -> list[dict[str, Any]]:
    """Search Semantic Scholar for research papers (optional API key for rate limits).

    Returns paper dicts; GitHub URLs are filled later via arXiv abstract when possible.
    Docs: https://api.semanticscholar.org/api-docs/
    """
    query = (query or "").strip()
    if not query:
        return []

    params: dict[str, Any] = {
        "query": query,
        "limit": int(max_results),
        "fields": "title,year,abstract,externalIds,url,venue,publicationTypes",
    }
    if min_year is not None:
        params["year"] = f"{int(min_year)}-"

    headers = {
        "User-Agent": _UA,
        "Accept": "application/json",
    }
    key = (os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY") or "").strip()
    if key:
        headers["x-api-key"] = key

    resp = _http_get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        headers=headers,
        timeout=30.0,
        retries=4,
        backoff_base=1.5,
        backoff_cap=60.0,
    )
    if resp is None:
        if not key:
            _note(
                "Semantic Scholar rate-limited (no API key). "
                "Set SEMANTIC_SCHOLAR_API_KEY for higher limits."
            )
        else:
            _note("Semantic Scholar request failed.")
        return []

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return []

    papers: list[dict[str, Any]] = []
    for item in data.get("data") or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        ext = item.get("externalIds") or {}
        arxiv_id = (ext.get("ArXiv") or "").strip()
        year = item.get("year")
        abstract = (item.get("abstract") or "") or ""
        blob = f"{title}\n{abstract}"
        papers.append(
            {
                "title": title,
                "arxiv_id": arxiv_id,
                "year": int(year) if year else None,
                "abstract": abstract[:2000],
                "paper_url": item.get("url")
                or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
                "keywords": [],
                "comment": "",
                "github_urls": extract_github_urls(blob),
                "venue": item.get("venue") or "",
                "source": "semantic_scholar",
            }
        )
    return papers


def _parse_arxiv_atom(xml: str) -> list[dict[str, Any]]:
    """Minimal Atom parse without requiring feedparser at import time for robustness."""
    try:
        import feedparser  # type: ignore

        feed = feedparser.parse(xml)
        papers: list[dict[str, Any]] = []
        for entry in feed.entries:
            title = re.sub(r"\s+", " ", (entry.get("title") or "")).strip()
            abstract = re.sub(r"\s+", " ", (entry.get("summary") or "")).strip()
            link = ""
            for l in entry.get("links") or []:
                if l.get("type") == "text/html" or l.get("rel") == "alternate":
                    link = l.get("href") or ""
                    break
            if not link:
                link = entry.get("id") or ""
            arxiv_id = ""
            m = _ARXIV_ID_RE.search(link or entry.get("id") or "")
            if m:
                arxiv_id = m.group(1)
            published = entry.get("published") or ""
            year = None
            if len(published) >= 4 and published[:4].isdigit():
                year = int(published[:4])
            cats = []
            for t in entry.get("tags") or []:
                term = t.get("term") or ""
                if term:
                    cats.append(term)
            comment = ""
            if hasattr(entry, "arxiv_comment"):
                comment = entry.arxiv_comment or ""
            blob = f"{title}\n{abstract}\n{comment}\n{link}"
            papers.append(
                {
                    "title": title,
                    "arxiv_id": arxiv_id,
                    "year": year,
                    "abstract": abstract,
                    "paper_url": link or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
                    "keywords": cats[:8],
                    "comment": comment,
                    "github_urls": extract_github_urls(blob),
                }
            )
        return papers
    except Exception:  # noqa: BLE001
        # Fallback: regex scrape of titles from Atom
        return _parse_arxiv_regex(xml)


def _parse_arxiv_regex(xml: str) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    entries = re.split(r"<entry>", xml)[1:]
    for chunk in entries:
        title_m = re.search(r"<title>(.*?)</title>", chunk, re.DOTALL)
        summary_m = re.search(r"<summary>(.*?)</summary>", chunk, re.DOTALL)
        id_m = re.search(r"<id>(.*?)</id>", chunk)
        published_m = re.search(r"<published>(.*?)</published>", chunk)
        title = re.sub(r"\s+", " ", (title_m.group(1) if title_m else "")).strip()
        abstract = re.sub(r"\s+", " ", (summary_m.group(1) if summary_m else "")).strip()
        link = (id_m.group(1) if id_m else "").strip()
        arxiv_id = ""
        m = _ARXIV_ID_RE.search(link)
        if m:
            arxiv_id = m.group(1)
        year = None
        pub = published_m.group(1) if published_m else ""
        if len(pub) >= 4 and pub[:4].isdigit():
            year = int(pub[:4])
        blob = f"{title}\n{abstract}\n{link}"
        papers.append(
            {
                "title": title,
                "arxiv_id": arxiv_id,
                "year": year,
                "abstract": abstract,
                "paper_url": link or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
                "keywords": [],
                "comment": "",
                "github_urls": extract_github_urls(blob),
            }
        )
    return papers


def find_github_for_title(title: str) -> str | None:
    """Fallback: web search for a GitHub repo matching a paper title."""
    title = (title or "").strip()
    if not title:
        return None
    results = web_search_raw(f'"{title}" site:github.com', max_results=8)
    title_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9]{4,}", title)}
    scored: list[tuple[int, str]] = []
    for r in results:
        url = r.get("url") or ""
        blob = f"{r.get('title', '')} {r.get('snippet', '')} {url}".lower()
        urls = extract_github_urls(url) + extract_github_urls(blob)
        for u in urls:
            # Skip obvious aggregators / meta lists
            low = u.lower()
            if any(
                bad in low
                for bad in (
                    "arxivtimes",
                    "papers-with-code",
                    "paperlist",
                    "awesome-",
                    "/awesome",
                )
            ):
                continue
            overlap = sum(1 for t in title_tokens if t in blob or t in low)
            scored.append((overlap, u))
    scored.sort(key=lambda x: (-x[0], x[1]))
    # Require at least two title-token overlaps to reduce unrelated repos
    for score, u in scored:
        if score >= 2:
            return u
    return None


def github_repo_info(repo_url: str) -> dict[str, Any] | None:
    """Fetch public repo metadata + README + dependency file presence via GitHub API."""
    urls = extract_github_urls(repo_url or "")
    if not urls:
        return None
    parsed = urlparse(urls[0])
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]

    api = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        with httpx.Client(timeout=25.0, headers=_github_headers(), follow_redirects=True) as client:
            r = _client_get_with_backoff(client, api)
            if r is None:
                return None
            if r.status_code == 404:
                return None
            r.raise_for_status()
            meta = r.json()

            readme_text = ""
            rr = _client_get_with_backoff(client, f"{api}/readme")
            if rr is not None and rr.status_code == 200:
                content_url = (rr.json() or {}).get("download_url")
                if content_url:
                    tr = _client_get_with_backoff(client, content_url)
                    if tr is not None and tr.status_code == 200:
                        readme_text = tr.text[:12000]

            def _has(path: str) -> bool:
                cr = _client_get_with_backoff(client, f"{api}/contents/{path}")
                return cr is not None and cr.status_code == 200

            def _fetch_text(path: str, limit: int = 4000) -> str:
                cr = _client_get_with_backoff(client, f"{api}/contents/{path}")
                if cr is None or cr.status_code != 200:
                    return ""
                data = cr.json()
                if isinstance(data, dict) and data.get("download_url"):
                    tr = _client_get_with_backoff(client, data["download_url"])
                    if tr is not None and tr.status_code == 200:
                        return tr.text[:limit]
                return ""

            files = {
                "requirements.txt": _has("requirements.txt"),
                "environment.yml": _has("environment.yml") or _has("environment.yaml"),
                "setup.py": _has("setup.py"),
                "pyproject.toml": _has("pyproject.toml"),
                "Dockerfile": _has("Dockerfile"),
                "README": bool(readme_text),
            }
            req_text = ""
            if files["requirements.txt"]:
                req_text = _fetch_text("requirements.txt")
            elif files["pyproject.toml"]:
                req_text = _fetch_text("pyproject.toml")

            # Collect .py paths from the default branch tree (for entrypoint validation)
            py_files: list[str] = []
            branch = meta.get("default_branch") or "main"
            tree_r = _client_get_with_backoff(
                client, f"{api}/git/trees/{branch}?recursive=1"
            )
            if tree_r is not None and tree_r.status_code == 200:
                for node in (tree_r.json() or {}).get("tree") or []:
                    if node.get("type") != "blob":
                        continue
                    path = node.get("path") or ""
                    if path.endswith(".py") and not path.startswith("."):
                        # Skip huge vendored trees
                        if any(
                            skip in path.replace("\\", "/")
                            for skip in ("/venv/", "/.venv/", "/site-packages/", "/node_modules/")
                        ):
                            continue
                        py_files.append(path)
                        if len(py_files) >= 80:
                            break

            return {
                "url": meta.get("html_url") or urls[0],
                "full_name": meta.get("full_name") or f"{owner}/{repo}",
                "description": meta.get("description") or "",
                "stars": int(meta.get("stargazers_count") or 0),
                "size_kb": int(meta.get("size") or 0),
                "default_branch": branch,
                "language": meta.get("language") or "",
                "license": ((meta.get("license") or {}) or {}).get("spdx_id") or "",
                "archived": bool(meta.get("archived")),
                "readme": readme_text,
                "files": files,
                "requirements_text": req_text,
                "py_files": py_files,
            }
    except Exception:  # noqa: BLE001
        return None
