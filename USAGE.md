# Usage guide — Paper-with-Code Reproduction Lab

How to run **agentFinalProject** locally and on Vercel.

## What this app does

Multi-agent pipeline (see `docs/final_project_multiagent_report.tex`):

1. **PaperFinder** — search **OpenAlex** (primary), then **arXiv** / **Semantic Scholar**; keep recent papers with author GitHub links; skip tutorials; **screen feasibility while searching**. If status shows rate-limit notes, wait a minute or set `SEMANTIC_SCHOLAR_API_KEY`.
2. **You pick** a paper from the screened list (verdict + entrypoint already shown)
3. **Feasibility report** loads on click (computed at search time)
4. **Permission** — you approve clone/install/run
5. **ExperimentRunner** — Docker sandbox (local only); streams **agent reasoning** live; archives under `runs/successful/` or `runs/failed/`
6. **ParamTweaker** — edit key params, re-run, compare

On a successful run, the agent also explains the paper in simple terms and what the entrypoint/arguments mean.

There is also a legacy **Research Scout** tab (single ReAct or Planner→Researcher→Writer→Critic).

### Hybrid deploy matrix

| Stage | Local | Vercel |
|-------|-------|--------|
| Find papers + feasibility | Yes | Yes |
| Docker clone / install / run | Yes | No — shows local commands |
| Param tweaks (re-run) | Yes | No |

## Prerequisites

- Python **3.10+**
- **OpenAI API key** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Docker Desktop** (for ExperimentRunner on your machine)
- **git** on PATH
- Optional: **GitHub token** for higher API rate limits

## Setup (Windows)

```powershell
cd agentFinalProject
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

You can either:

- Paste your key in the app’s **OpenAI API key** password field (recommended; never written to disk), or  
- Put `OPENAI_API_KEY=sk-...` in `.env` as a local fallback.

Optional `.env` knobs:

```
OPENAI_MODEL=gpt-4o-mini
MIN_PAPER_YEAR=2021
# GITHUB_TOKEN=ghp_...
# SEMANTIC_SCHOLAR_API_KEY=...
# RUN_TIMEOUT_SEC=600
GRADIO_SERVER_PORT=7860
```

## Run locally

```powershell
.\.venv\Scripts\activate
python app.py
```

Open **http://127.0.0.1:7860** — paste your OpenAI key in the secure field at the top.

### Reproduction Lab walkthrough

1. Paste **OpenAI API key** (masked).  
2. Open the **Reproduction Lab** tab.  
3. Enter a research topic → **Find runnable papers**.  
4. Click a row — feasibility (verdict + entrypoint) was already screened at search time.  
5. If verdict is not `IMPOSSIBLE`, check **I approve clone / install / run**.  
6. Click **Run experiment** (first run may pull `python:3.11-slim`).  
7. Edit the **Parameters JSON** if shown → **Re-run with tweaks**.  
8. Finished runs land under `runs/successful/` or `runs/failed/`.

### Research Scout tab

Same as the earlier scout: pick Single or Team mode, enter a topic, run, optionally save markdown under `outputs/`. Uses the same secure API key field.

## Deploy on Vercel

1. Push **this folder** (`agentFinalProject/`) to GitHub (do **not** commit `.env`).  
2. In Vercel: **Add New Project** → import the repo → set **Root Directory** to `agentFinalProject` if the repo is a monorepo.  
3. Framework preset: **Other**. Vercel will detect the FastAPI/`app` ASGI entry in `app.py`.  
4. Environment variables (Project → Settings → Environment Variables):  
   - Optional fallback: `OPENAI_API_KEY` (users can still paste their own key in the UI)  
   - Optional: `OPENAI_MODEL`, `GITHUB_TOKEN`, `SEMANTIC_SCHOLAR_API_KEY`, `MIN_PAPER_YEAR`  
5. Deploy. `vercel.json` sets `maxDuration` to **300s** and adds basic security headers.

**Hybrid behavior on Vercel**

| Stage | Works on Vercel? |
|-------|------------------|
| Paste API key / search / feasibility | Yes |
| Docker clone / install / run | No — shows local commands |
| Param tweaks (re-run) | No |

**API key security**

- UI field is `type=password` (masked).  
- Key is held in a **request-scoped context var** only (not saved to `.env`, `runs/`, or report files).  
- Status text is redacted if a key ever appears in an error string.  
- Prefer HTTPS (Vercel default). Do not share screen recordings that show the paste field before it’s masked.

On Vercel, **Run experiment** will not execute Docker; use your machine for reproduction runs.

## Project layout

```
agentFinalProject/
  app.py                 # Gradio + FastAPI (Vercel ASGI entry)
  USAGE.md               # this file
  FINAL_PROJECT.md
  requirements.txt
  vercel.json
  scout/                 # single ReAct agent
  scout_team/            # research team scaffold
  replab/                # paper-with-code lab
    schemas.py
    tools.py             # arXiv + GitHub
    finder.py
    analyst.py
    runner.py            # Docker runner
    tweaker.py
  docs/
    final_project_multiagent_report.tex
    agentsResearch.txt
  runs/                  # local clones + logs (gitignored)
  outputs/               # saved scout reports
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Missing API key | Paste key in the secure UI field, or set `OPENAI_API_KEY` in `.env` / Vercel env |
| No candidates | Broader query; or add `GITHUB_TOKEN` if rate-limited |
| Docker errors | Start Docker Desktop; run `docker pull python:3.11-slim` |
| Clone fails | Check `git` and network; repo must be public |
| Run times out | Raise `RUN_TIMEOUT_SEC` or pick a smaller demo entrypoint |
| Vercel timeout | Keep search queries small; run Docker steps locally |

## Safety notes

- Only public GitHub repos are used.  
- `IMPOSSIBLE` verdicts refuse to run.  
- Docker runs are capped (`--memory 4g --cpus 2`) with a wall-clock timeout.  
- You must explicitly grant permission before any clone/install/run.  
- OpenAI keys pasted in the UI are **not** written to disk; prefer the password field on shared/Vercel deploys.
