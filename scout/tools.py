"""Free web search and page fetch — LangChain tools the agent can call."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool

_SEARCH_DELAY_SEC = 1.0
_FETCH_TIMEOUT = 15.0
_USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchScout/1.0; +https://localhost) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _chars_per_page() -> int:
    return int(os.getenv("CHARS_PER_PAGE", "4000"))


def _is_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:  # noqa: BLE001
        return False


def web_search_raw(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web via ddgs (no API key). Returns title/url/snippet dicts."""
    query = (query or "").strip()
    if not query:
        return []

    results: list[dict[str, str]] = []
    last_err: Exception | None = None

    for attempt in range(2):
        try:
            with DDGS() as ddgs:
                try:
                    raw = list(
                        ddgs.text(query, max_results=max_results, backend="duckduckgo")
                    )
                except Exception:
                    raw = list(ddgs.text(query, max_results=max_results))
            for item in raw:
                url = (item.get("href") or item.get("url") or "").strip()
                title = (item.get("title") or "").strip()
                snippet = (item.get("body") or item.get("snippet") or "").strip()
                if url and _is_http_url(url):
                    results.append({"title": title or url, "url": url, "snippet": snippet})
            time.sleep(_SEARCH_DELAY_SEC)
            return results
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(_SEARCH_DELAY_SEC * (attempt + 2))

    if last_err:
        return []
    return results


def fetch_page_raw(url: str, max_chars: int | None = None) -> dict[str, str] | None:
    """Fetch a URL and extract main article text. Returns None on failure."""
    if max_chars is None:
        max_chars = _chars_per_page()
    url = (url or "").strip()
    if not _is_http_url(url):
        return None

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception:  # noqa: BLE001
        return None

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        url=url,
    )
    if not extracted or not extracted.strip():
        return None

    text = extracted.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"

    meta = trafilatura.extract_metadata(html)
    page_title = ""
    if meta is not None:
        page_title = (getattr(meta, "title", None) or "").strip()

    return {"url": url, "title": page_title or url, "text": text}


@tool
def web_search(query: str) -> str:
    """Search the public web for information on a query.

    Use this to discover relevant articles, docs, and news. Returns a JSON list of
    results with title, url, and snippet. After searching, call fetch_page on the
    most promising URLs to read full content.
    """
    hits = web_search_raw(query, max_results=5)
    if not hits:
        return json.dumps({"error": "No results (empty or rate-limited). Try a different query."})
    return json.dumps(hits, ensure_ascii=False)


@tool
def fetch_page(url: str) -> str:
    """Download a web page and extract its main readable text.

    Pass a full http(s) URL from web_search results. Returns JSON with title, url,
    and text (truncated). Skip paywalled or failed pages and try another URL.
    """
    page = fetch_page_raw(url)
    if not page:
        return json.dumps(
            {
                "error": "Could not fetch or extract text from this URL.",
                "url": url,
            }
        )
    return json.dumps(page, ensure_ascii=False)


AGENT_TOOLS = [web_search, fetch_page]


def sources_from_tool_content(tool_name: str, content: str) -> list[dict[str, Any]]:
    """Parse tool JSON into SourceDoc-like dicts for the UI / final report."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict) and data.get("error"):
        return []

    if tool_name == "web_search" and isinstance(data, list):
        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "text": item.get("snippet", ""),
            }
            for item in data
            if isinstance(item, dict) and item.get("url")
        ]

    if tool_name == "fetch_page" and isinstance(data, dict) and data.get("url"):
        return [
            {
                "title": data.get("title", ""),
                "url": data.get("url", ""),
                "snippet": "",
                "text": data.get("text", ""),
            }
        ]

    return []
