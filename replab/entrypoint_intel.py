"""Inspect entrypoint scripts before running — argparse, README demos, safe commands."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from replab.llm import llm

_PLACEHOLDER_RE = re.compile(r"<[^>\s]+>|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")


@dataclass
class CliArg:
    name: str
    kind: str  # "positional" | "optional"
    required: bool
    default: Any = None
    choices: list[str] = field(default_factory=list)
    help: str = ""


@dataclass
class ScriptCliSpec:
    script: str
    source: str
    docstring: str = ""
    args: list[CliArg] = field(default_factory=list)
    uses_argparse: bool = False
    uses_click: bool = False
    has_main: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def required_positionals(self) -> list[CliArg]:
        return [a for a in self.args if a.kind == "positional" and a.required]


@dataclass
class EntrypointPlan:
    command: str
    script: str
    spec: ScriptCliSpec | None = None
    filled_args: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        return None


def _call_kwargs(node: ast.Call) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg:
            out[kw.arg] = _literal(kw.value)
    return out


def parse_script_cli(source: str, script: str = "") -> ScriptCliSpec:
    """Statically inspect a Python entrypoint for CLI expectations."""
    spec = ScriptCliSpec(script=script, source=source or "")
    if not source.strip():
        spec.notes.append("Empty script source.")
        return spec

    # Module docstring
    try:
        tree = ast.parse(source)
        spec.docstring = ast.get_docstring(tree) or ""
        spec.has_main = any(
            isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            # rough: if __name__ == "__main__"
            for n in tree.body
        ) or ('__name__ == "__main__"' in source) or ("__name__=='__main__'" in source)
    except SyntaxError:
        spec.notes.append("Could not AST-parse script; falling back to regex.")
        tree = None

    spec.uses_argparse = "argparse" in source or "ArgumentParser" in source
    spec.uses_click = "import click" in source or "@click." in source

    if tree is not None:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_add = isinstance(func, ast.Attribute) and func.attr == "add_argument"
            if not is_add:
                continue
            if not node.args:
                continue
            first = _literal(node.args[0])
            if not isinstance(first, str):
                continue
            kwargs = _call_kwargs(node)
            optional = first.startswith("-")
            nargs = kwargs.get("nargs")
            required_kw = kwargs.get("required")
            default = kwargs.get("default")
            choices = kwargs.get("choices") or []
            if isinstance(choices, (list, tuple)):
                choice_list = [str(c) for c in choices]
            else:
                choice_list = []

            if optional:
                # --flag: required only if required=True
                req = bool(required_kw) if required_kw is not None else False
                kind = "optional"
                name = first.lstrip("-")
            else:
                kind = "positional"
                # nargs='?' or '*' => not required
                if nargs in ("?", "*"):
                    req = False
                elif required_kw is False:
                    req = False
                else:
                    req = default is None
                name = first

            help_txt = kwargs.get("help") if isinstance(kwargs.get("help"), str) else ""
            spec.args.append(
                CliArg(
                    name=name,
                    kind=kind,
                    required=req,
                    default=default,
                    choices=choice_list,
                    help=help_txt or "",
                )
            )

    # Click arguments (lightweight)
    if spec.uses_click and not spec.args:
        for m in re.finditer(
            r"@click\.(argument|option)\(\s*['\"]([^'\"]+)['\"]([^)]*)\)",
            source,
        ):
            kind_raw, name, rest = m.group(1), m.group(2), m.group(3)
            optional = kind_raw == "option" or name.startswith("-")
            req = "required=True" in rest.replace(" ", "")
            if not optional:
                req = "required=False" not in rest.replace(" ", "")
            spec.args.append(
                CliArg(
                    name=name.lstrip("-"),
                    kind="optional" if optional else "positional",
                    required=req if optional else True,
                )
            )

    return spec


def read_repo_docs(repo_dir: Path, limit: int = 40000) -> str:
    chunks: list[str] = []
    for rel in (
        "README.md",
        "readme.md",
        "README.rst",
        "data/README.md",
        "docs/usage.md",
        "docs/README.md",
        "CONTRIBUTING.md",
    ):
        path = repo_dir / rel
        if not path.is_file():
            continue
        try:
            chunks.append(f"===== {rel} =====\n" + path.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            continue
        if sum(len(c) for c in chunks) >= limit:
            break
    return "\n\n".join(chunks)[:limit]


def example_values_from_docs(docs: str, n: int = 8) -> list[str]:
    """Concrete numeric / path-like demo tokens from docs (no placeholders)."""
    docs = docs or ""
    # load_scene(a, b, c, d)
    m = re.search(
        r"load_scene\(\s*"
        + r",\s*".join([r"([-+]?\d+(?:\.\d+)?)"] * min(n, 4))
        + r"\s*[,)]",
        docs,
    )
    if m:
        return [m.group(i) for i in range(1, m.lastindex + 1)]

    m = re.search(
        r"python3?\s+[\w./-]+\.py\s+"
        + r"\s+".join([r"([-+]?\d+(?:\.\d+)?)"] * min(n, 4)),
        docs,
        flags=re.IGNORECASE,
    )
    if m:
        return [m.group(i) for i in range(1, m.lastindex + 1)]

    for row in re.finditer(
        r"\|\s*[^|]+\s*\|\s*"
        + r"\|\s*".join([r"([-+]?\d+(?:\.\d+)?)"] * 4)
        + r"\s*\|",
        docs,
    ):
        return [row.group(i) for i in range(1, 5)]

    # Quoted paths that look like sample data
    paths = re.findall(r"['\"](data/[\w./-]+\.(?:xml|json|csv|pt|pth|npz|txt))['\"]", docs)
    return paths[:n]


def _is_flag_token(tok: str) -> bool:
    """True for -h/--flag, but not numeric literals like -97.74."""
    if tok.startswith("--"):
        return True
    if tok.startswith("-") and re.match(r"^-\d", tok):
        return False
    if tok.startswith("-") and re.match(r"^-[A-Za-z]", tok):
        return True
    return False


def _tokens_after_script(cmd: str) -> list[str]:
    parts = (cmd or "").split()
    for i, tok in enumerate(parts):
        if tok.endswith(".py"):
            return [t for t in parts[i + 1 :] if not _PLACEHOLDER_RE.search(t)]
    return []


def _positional_tokens(tokens: list[str]) -> list[str]:
    provided: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_flag_token(tok):
            if i + 1 < len(tokens) and not _is_flag_token(tokens[i + 1]):
                i += 2
            else:
                i += 1
            continue
        provided.append(tok)
        i += 1
    return provided


def _script_from_cmd(cmd: str) -> str | None:
    m = re.search(
        r"(?:python(?:3)?|streamlit\s+run)\s+([^\s]+\.py)",
        cmd or "",
        flags=re.I,
    )
    return m.group(1).replace("\\", "/").lstrip("./") if m else None


def _is_streamlit_source(source: str) -> bool:
    return bool(
        re.search(r"(?m)^\s*(import streamlit|from streamlit\b)", source or "")
    )


def _has_placeholder_data_path(source: str) -> bool:
    blob = source or ""
    return bool(
        re.search(
            r"DATA_FILEPATH\s*=\s*[\"']data file path[\"']|"
            r"[\"']/?path/to/[^\"']+[\"']|"
            r"[\"']your[_ -]?data[\"']|"
            r"[\"']TODO[:\s].*?\.(?:csv|parquet|json)[\"']",
            blob,
            flags=re.I,
        )
    )


def _find_repo_csv(repo_dir: Path) -> str | None:
    preferred: list[str] = []
    other: list[str] = []
    for p in repo_dir.rglob("*.csv"):
        try:
            rel = p.relative_to(repo_dir).as_posix()
        except ValueError:
            continue
        if any(skip in f"/{rel}/" for skip in ("/venv/", "/.venv/", "/site-packages/")):
            continue
        if "data" in rel.lower() or "final" in rel.lower():
            preferred.append(rel)
        else:
            other.append(rel)
    return (preferred or other or [None])[0]


def _find_import_smoke(repo_dir: Path) -> str | None:
    """Non-UI smoke test when Streamlit demos aren't runnable headlessly."""
    for rel in (
        "ChronosBolt/chronos_bolt_adapter.py",
        "chronos_bolt_adapter.py",
    ):
        if (repo_dir / rel).is_file():
            parent = str(Path(rel).parent).replace("\\", "/")
            mod = Path(rel).stem
            return (
                f"export PYTHONPATH=/work/{parent}:/work:${{PYTHONPATH:-}}; "
                f"python -c \"import {mod}; print('smoke_ok', {mod}.__file__)\""
            )
    # Generic: import the package dir if present
    for pkg in ("src", "lib"):
        if (repo_dir / pkg / "__init__.py").is_file():
            return (
                "export PYTHONPATH=/work:${PYTHONPATH:-}; "
                f"python -c \"import {pkg}; print('smoke_ok')\""
            )
    return None


def _streamlit_run_cmd(script: str) -> str:
    return (
        f"streamlit run {script} "
        "--server.headless true --server.port 8501 "
        "--browser.gatherUsageStats false"
    )


def _patch_data_filepath_preamble(script_rel: str, csv_rel: str) -> str:
    """Shell snippet: rewrite placeholder DATA_FILEPATH in the entry script."""
    # Use python so we don't fight shell quoting on Windows-mounted paths
    return (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"p = Path({script_rel!r})\n"
        "t = p.read_text(encoding='utf-8')\n"
        "import re\n"
        f"csv = {csv_rel!r}\n"
        "t2, n = re.subn(\n"
        "    r'DATA_FILEPATH\\s*=\\s*[\\'\\\"][^\\'\\\"]*[\\'\\\"]',\n"
        "    f'DATA_FILEPATH = \"{csv}\"',\n"
        "    t,\n"
        "    count=1,\n"
        ")\n"
        "if n:\n"
        "    p.write_text(t2, encoding='utf-8')\n"
        "    print(f'patched DATA_FILEPATH -> {csv}')\n"
        "else:\n"
        "    print('DATA_FILEPATH not patched')\n"
        "PY\n"
    )


def _fill_required_args(
    spec: ScriptCliSpec,
    existing_tokens: list[str],
    docs: str,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Return positional tokens to append, mapping, and notes."""
    notes: list[str] = []
    filled: dict[str, str] = {}
    required = spec.required_positionals
    if not required:
        return [], filled, notes

    provided = _positional_tokens(existing_tokens)

    missing = required[len(provided) :]
    if not missing:
        return [], filled, notes

    demos = example_values_from_docs(docs, n=max(8, len(missing) + 2))
    notes.append(
        f"Script requires positionals {[a.name for a in required]}; "
        f"missing {[a.name for a in missing]}."
    )

    # Name-aware heuristics
    name_hints: dict[str, str] = {}
    for a in missing:
        if a.choices:
            name_hints[a.name] = a.choices[0]
        elif a.default is not None:
            name_hints[a.name] = str(a.default)

    # Map common geo bbox names onto demo floats
    geo_names = (
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "min_lat",
        "max_lat",
        "min_lon",
        "max_lon",
    )
    geo_demos = [d for d in demos if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", d)]
    if len(geo_demos) >= 4 and all(
        a.name.lower() in geo_names
        or "lat" in a.name.lower()
        or "lon" in a.name.lower()
        for a in missing
    ):
        for a, val in zip(missing, geo_demos):
            name_hints[a.name] = val

    # Generic: consume remaining demo numbers/paths in order
    demo_i = 0
    extras: list[str] = []
    for a in missing:
        if a.name in name_hints:
            val = name_hints[a.name]
        else:
            while demo_i < len(demos) and demos[demo_i] in name_hints.values():
                demo_i += 1
            if demo_i < len(demos):
                val = demos[demo_i]
                demo_i += 1
            else:
                # Last-resort tiny defaults by name
                if "seed" in a.name.lower():
                    val = "0"
                elif "epoch" in a.name.lower() or "step" in a.name.lower():
                    val = "1"
                elif "path" in a.name.lower() or "file" in a.name.lower() or "dir" in a.name.lower():
                    val = "."
                elif "lat" in a.name.lower() or "lon" in a.name.lower():
                    val = "0.0"
                else:
                    notes.append(f"No demo value found for required arg `{a.name}`.")
                    continue
        filled[a.name] = val
        extras.append(val)

    return extras, filled, notes


def _llm_complete_command(
    script: str,
    source: str,
    docs: str,
    partial_cmd: str,
    spec: ScriptCliSpec,
) -> str | None:
    """Ask the LLM for a concrete command once static fill is incomplete."""
    req = ", ".join(a.name for a in spec.required_positionals) or "(none)"
    try:
        resp = llm().invoke(
            [
                SystemMessage(
                    content=(
                        "You prepare a single safe smoke-test command for a cloned paper repo at /work.\n"
                        "Read the entrypoint source and docs carefully.\n"
                        "Return ONLY one bash command line (no markdown).\n"
                        "Rules:\n"
                        "- Use the real script path given.\n"
                        "- Fill every required CLI argument with concrete demo values (numbers/paths from docs).\n"
                        "- Never use placeholders like <lat_min>, $VAR, or other_script.py.\n"
                        "- Prefer the smallest/shortest demo (tiny bbox, 1 epoch, --help only if nothing else works).\n"
                        "- Do not download huge datasets if a smaller example exists in-repo.\n"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Script: {script}\n"
                        f"Required positionals: {req}\n"
                        f"Partial command: {partial_cmd}\n\n"
                        f"Entrypoint source (truncated):\n{source[:6000]}\n\n"
                        f"Repo docs (truncated):\n{docs[:5000]}\n"
                    )
                ),
            ]
        )
        text = str(resp.content or "").strip()
        text = re.sub(r"^```(?:bash|sh)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
        text = text.splitlines()[0].strip() if text else ""
        if not text or _PLACEHOLDER_RE.search(text):
            return None
        if "python" not in text.lower():
            return None
        return text
    except Exception:  # noqa: BLE001
        return None


def _resolve_script_in_repo(script: str, repo_dir: Path) -> str | None:
    script = (script or "").replace("\\", "/").lstrip("./")
    if not script:
        return None
    direct = repo_dir / script
    if direct.is_file():
        return script
    base = Path(script).name
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
    m = re.search(
        r"((?:python(?:3)?|streamlit\s+run)\s+)([^\s]+\.py)",
        cmd or "",
        flags=re.IGNORECASE,
    )
    if not m:
        return cmd
    resolved = _resolve_script_in_repo(m.group(2), repo_dir)
    if not resolved:
        return cmd
    return cmd[: m.start(2)] + resolved + cmd[m.end(2) :]


def prepare_entrypoint_command(
    repo_dir: Path,
    suggested: str,
    *,
    use_llm: bool = True,
) -> EntrypointPlan:
    """Read the entrypoint script + docs, then build a runnable command.

    This is the preflight step before Docker: understand argparse / required args
    and fill demo values so placeholder or incomplete commands are not executed.
    """
    from replab.analyst import _sanitize_entrypoint_cmd

    notes: list[str] = []
    cmd = _sanitize_entrypoint_cmd(suggested or "")
    cmd = _rewrite_cmd_script_paths(cmd, repo_dir)
    if not cmd:
        cmd = "python -c \"print('no entrypoint')\""

    script_rel = _script_from_cmd(cmd)
    if not script_rel:
        return EntrypointPlan(command=cmd, script="", notes=["No .py entrypoint to inspect."])

    resolved = _resolve_script_in_repo(script_rel, repo_dir) or script_rel
    if resolved != script_rel:
        cmd = cmd.replace(script_rel, resolved, 1)
        script_rel = resolved
        notes.append(f"Resolved script path to {script_rel}.")

    script_path = repo_dir / script_rel
    source = ""
    if script_path.is_file():
        try:
            source = script_path.read_text(encoding="utf-8", errors="replace")[:24000]
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Could not read script: {exc}")
    else:
        notes.append(f"Script not found on disk: {script_rel}")
        return EntrypointPlan(command=cmd, script=script_rel, notes=notes)

    spec = parse_script_cli(source, script=script_rel)
    docs = read_repo_docs(repo_dir)

    # Streamlit apps must not be executed with bare `python app.py`
    if _is_streamlit_source(source):
        notes.append(
            "Detected Streamlit UI — bare `python` lacks ScriptRunContext; "
            "using `streamlit run` (or a non-UI smoke test)."
        )
        csv = _find_repo_csv(repo_dir)
        if _has_placeholder_data_path(source) and not csv:
            smoke = _find_import_smoke(repo_dir)
            if smoke:
                notes.append(
                    "Streamlit demo has a placeholder data path and no CSV in the clone; "
                    "falling back to a library import smoke test."
                )
                return EntrypointPlan(
                    command=smoke,
                    script=script_rel,
                    spec=spec,
                    notes=notes,
                )
            notes.append(
                "Cannot smoke-test this Streamlit demo: placeholder DATA_FILEPATH "
                "and dataset files are missing from the repo clone."
            )
            return EntrypointPlan(
                command=(
                    "python -c \"raise SystemExit("
                    "'Streamlit demo needs a real dataset path "
                    "(DATA_FILEPATH is still a placeholder and no CSV was found).'"
                    ")\""
                ),
                script=script_rel,
                spec=spec,
                notes=notes,
            )

        preamble = ""
        if _has_placeholder_data_path(source) and csv:
            notes.append(f"Will patch placeholder DATA_FILEPATH to `{csv}` before launch.")
            preamble = _patch_data_filepath_preamble(script_rel, csv)

        # WaterRAF-style demos look for adapter beside the Streamlit app
        adapter = repo_dir / "ChronosBolt" / "chronos_bolt_adapter.py"
        if adapter.is_file():
            app_dir = str(Path(script_rel).parent).replace("\\", "/")
            notes.append(
                "Copying ChronosBolt adapter next to the Streamlit app so imports resolve."
            )
            preamble += (
                f"mkdir -p '{app_dir}/Retrieval-Augmented-Time-Series-Forecasting' && "
                f"cp -f ChronosBolt/chronos_bolt_adapter.py "
                f"'{app_dir}/Retrieval-Augmented-Time-Series-Forecasting/' && "
                f"cp -f ChronosBolt/chronos_bolt_adapter.py '{app_dir}/' && "
            )

        cmd = (
            "export PYTHONPATH=/work/ChronosBolt:/work:${PYTHONPATH:-}; "
            + preamble
            + _streamlit_run_cmd(script_rel)
        )
        return EntrypointPlan(
            command=cmd,
            script=script_rel,
            spec=spec,
            filled_args={"DATA_FILEPATH": csv} if csv else {},
            notes=notes,
        )

    existing = _tokens_after_script(cmd)

    extras, filled, fill_notes = _fill_required_args(spec, existing, docs)
    notes.extend(fill_notes)
    if extras:
        cmd = f"{cmd} {' '.join(extras)}".strip()
        notes.append(f"Filled args from docs/heuristics: {filled}")

    provided_n = len(_positional_tokens(_tokens_after_script(cmd)))
    if use_llm and len(spec.required_positionals) > provided_n:
        notes.append("Static fill incomplete — asking LLM to complete command from script+docs.")
        alt = _llm_complete_command(script_rel, source, docs, cmd, spec)
        if alt:
            alt = _sanitize_entrypoint_cmd(alt)
            alt = _rewrite_cmd_script_paths(alt, repo_dir)
            if alt and not _PLACEHOLDER_RE.search(alt):
                cmd = alt
                notes.append(f"LLM-completed command: {cmd}")

    # Final safety: strip any leftover placeholders
    if _PLACEHOLDER_RE.search(cmd):
        notes.append("Removed leftover placeholders from command.")
        cmd = _sanitize_entrypoint_cmd(cmd)

    provided_n = len(_positional_tokens(_tokens_after_script(cmd)))
    if spec.required_positionals and provided_n < len(spec.required_positionals):
        notes.append(
            "Required CLI args still incomplete; falling back to --help smoke check."
        )
        cmd = f"python {script_rel} --help"

    return EntrypointPlan(
        command=cmd,
        script=script_rel,
        spec=spec,
        filled_args=filled,
        notes=notes,
    )
