# Overleaf: RePLab scientific short paper

CHAINS-style final project report for the Paper-with-Code Reproduction Lab.

## Upload to Overleaf

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project** → **Upload Project**.
2. Zip the `overleaf/` folder (must contain `main.tex` at the zip root), or create a blank project and upload `main.tex`.
3. Set the main document to `main.tex` and **Recompile**.

## Before you submit / share

- Edit the `\author{...}` block (name, university, email).
- Optionally add co-authors / advisor lines.
- If you later run the evaluation protocol, add a Results table and cite real numbers only.

## Local compile (optional)

```bash
pdflatex main.tex
pdflatex main.tex
```

## Related docs

- Design brief (earlier): `../final_project_multiagent_report.tex`
- Product usage: `../../USAGE.md`
