# Final Project — Paper-with-Code Reproduction Lab

Implemented pipeline (see LaTeX brief + **USAGE.md**):

1. Retrieve papers that have **public code** (arXiv + GitHub validation)
2. User **chooses** by title / keywords
3. **Summarize** + hardware/software needs + alarms
4. **IMPOSSIBLE** stops; soft limits ask **permission**
5. Sandboxed **Docker** reproduction (local); Vercel shows local commands
6. **Parameter tweaking** + comparison

```
PaperFinder → USER pick → FeasibilityAnalyst
                              ├─ IMPOSSIBLE → stop
                              └─ RISKY/READY → USER permission
                                   → ExperimentRunner (Docker, local)
                                   → ParamTweaker
```

## Quick start

See **[USAGE.md](USAGE.md)** for setup, env vars, local walkthrough, Docker, and Vercel.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# paste OPENAI_API_KEY
python app.py
```

## Code map

| Path | Role |
|------|------|
| `replab/` | Paper-with-code agents |
| `scout/` | Original single ReAct research agent |
| `scout_team/` | Early Planner/Researcher/Writer/Critic scaffold |
| `app.py` | Gradio UI (Reproduction Lab + Research Scout tabs) |
| `docs/final_project_multiagent_report.tex` | Design brief |
| `docs/overleaf/main.tex` | CHAINS-style scientific short paper (Overleaf) |
