# Paper-with-Code Reproduction Lab

Multi-agent Gradio app built with **LangGraph** + **OpenAI**.

- **Reproduction Lab (final):** find arXiv papers with public GitHub code → you pick → feasibility report → permission → Docker run → param tweaks. **The Docker run and param tweaks are localhost-only**; the Vercel deployment serves search + feasibility and disables those stages.
- **Research Scout:** single ReAct agent or Planner → Researcher → Writer → Critic team.

**Start here:** [USAGE.md](USAGE.md) · Overleaf paper: [docs/overleaf/main.tex](docs/overleaf/main.tex) · design brief: [docs/final_project_multiagent_report.tex](docs/final_project_multiagent_report.tex)

## Quick start

```powershell
cd agentFinalProject
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# optional: set OPENAI_API_KEY=sk-...  (or paste in the app UI)
python app.py
```

Open **http://127.0.0.1:7860**, paste your OpenAI key in the secure field. For the runner stage, start **Docker Desktop**.

## Pipeline

```
PaperFinder → USER pick → FeasibilityAnalyst
                              ├─ IMPOSSIBLE → stop
                              └─ RISKY/READY → USER permission
                                   → ExperimentRunner (Docker, local)
                                   → ParamTweaker
```

## Layout

```
app.py            Gradio + FastAPI (Vercel entry)
replab/           paper-with-code agents
scout/            single ReAct scout
scout_team/       research team scaffold
docs/             LaTeX brief + research notes
USAGE.md          how to run & deploy
```

## Vercel

Deploy this folder; users can paste an API key in the UI (preferred) and/or set `OPENAI_API_KEY` in project env vars. `vercel.json` allows up to 300s and adds basic security headers.

### Reproduction runs are localhost-only

Vercel's serverless functions have no Docker daemon, no `git`, and no writable
project workspace, so **ExperimentRunner** and **ParamTweaker** cannot run there.
The deployed site shows a banner and disables those controls.

| Stage | Vercel | Localhost |
|---|---|---|
| Find papers with code | Yes | Yes |
| Feasibility report | Yes | Yes |
| Clone / install / run in Docker | No | Yes |
| Parameter tweaks + re-run | No | Yes |

Run `python app.py` on your own machine (with Docker Desktop started) for the
full pipeline.

### Secrets

Never commit `.env` — it is git-ignored. Set `OPENAI_API_KEY` in Vercel
project settings, or let each user paste their own key into the masked UI field.
