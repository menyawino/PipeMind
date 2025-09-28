from __future__ import annotations
import os
import json
import typer
from typing import Optional
from rich import print

from pipemind.registry.parser import parse_workflow_to_registry
from pipemind.dag.builder import build_dag_for_goal
from pipemind.cli.intake import required_fields_for_goal, interactive_collect
from pipemind.tools.generate_wrappers import generate as gen_wrappers
from pipemind.utils.llm import chat as llm_chat
from pipemind.snakemake.generator import materialize_and_optionally_run
import json as _json
import re as _re
from pipemind.snakemake.generator import materialize_and_optionally_run


app = typer.Typer(add_completion=False, help="PipeMind CLI")


@app.command()
def parse(
    workflow_dir: str = typer.Argument(..., help="Path to Snakemake workflow directory"),
    out_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Output registry YAML path"),
    keep_shell_comments: bool = typer.Option(True, help="Preserve all shell text including comments in commands"),
):
    """Parse Snakemake rules into a typed registry YAML."""
    reg = parse_workflow_to_registry(workflow_dir, out_yaml, keep_shell_comments=keep_shell_comments)
    print(f"[green]Wrote registry with {len(reg.tools)} tools to {out_yaml}")


@app.command()
def serve(
    registry_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Registry YAML path"),
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Start FastAPI + FastMCP server exposing schemas and MCP tools."""
    # Lazy import to avoid requiring fastmcp unless serving
    from pipemind.mcp_server.server import create_app
    api, mcp = create_app(registry_yaml)
    import uvicorn
    uvicorn.run(api, host=host, port=port)


@app.command()
def goal(
    registry_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Registry YAML path"),
    target_output: str = typer.Argument(..., help="Desired final output path template (with wildcards) or io type:prefix"),
    fill: Optional[str] = typer.Option(None, help="JSON string of known wildcard/param values"),
):
    """Build a dynamic DAG to achieve target_output using I/O matching."""
    known = json.loads(fill) if fill else {}
    plan = build_dag_for_goal(registry_yaml, target_output, known)
    print(json.dumps(plan, indent=2))


@app.command()
def intake(
    registry_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Registry YAML path"),
    target_output: str = typer.Argument(..., help="Desired final output path template"),
):
    fields = required_fields_for_goal(registry_yaml, target_output)
    vals = interactive_collect(fields)
    print(json.dumps(vals, indent=2))


@app.command()
def run(
    target: str = typer.Argument(..., help="Snakemake target to build (path)"),
    workflow_dir: str = typer.Argument(..., help="Path to Snakemake workflow directory"),
    snakefile: str = typer.Option(None, help="Snakefile path (defaults to <workflow_dir>/Snakefile)"),
    cores: int = typer.Option(4, help="Cores for snakemake"),
):
    """Run Snakemake for the given target."""
    import subprocess
    snakefile_path = snakefile if snakefile else os.path.join(workflow_dir, "Snakefile")
    cmd = ["snakemake", "-s", snakefile_path, target, "-c", str(cores), "--rerun-incomplete", "--printshellcmds"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("[red]Snakemake failed:\n" + res.stderr)
        raise typer.Exit(code=res.returncode)


@app.command()
def wrappers(
    registry_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Registry YAML path"),
    out_dir: str = typer.Option("pipemind/tools/generated", help="Output directory for wrappers"),
):
    n = gen_wrappers(registry_yaml, out_dir)
    print(f"Generated {n} wrappers in {out_dir}")


@app.command()
def llm(
    prompt: str = typer.Argument(..., help="User prompt"),
    model: str | None = typer.Option(None, help="OpenAI model id (defaults from env)"),
    base_url: str | None = typer.Option(None, help="Override base URL (OPENAI_BASE_URL)"),
    api_key: str | None = typer.Option(None, help="Override API key (OPENAI_API_KEY/OPENAI_API/OPENAI_KEY)"),
    provider: str | None = typer.Option(None, help="Force provider: 'ollama' or 'openai' (overrides env heuristics)"),
):
    """Send a quick chat to the configured OpenAI-compatible API."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant for bioinformatics pipelines."},
        {"role": "user", "content": prompt},
    ]
    try:
        out = llm_chat(msgs, model=model, api_key=api_key, base_url=base_url, provider=provider)
        if isinstance(out, tuple):  # (text, meta) from return_meta future usage
            out = out[0]
    except Exception as e:
        print(f"[red]LLM error:[/red] {e}")
        raise typer.Exit(code=1)
    if not out or not out.strip():
        print("[yellow]LLM returned an empty message. Check model/base URL/API key or permissions.[/yellow]")
    else:
        print(out)


def _extract_plan(text: str) -> tuple[list[str], dict[str, str]]:
    """Attempt to extract a JSON plan from model output.

    Accepts either a raw JSON object or fenced ```json blocks containing keys:
      {
        "goal_outputs": ["path/or/io_type:template"],
        "known": {"wildcard": "value"}
      }
    Returns (goal_outputs, known). Empty list/dict if not found.
    """
    if not text:
        return [], {}
    blocks = []
    fence_pat = _re.compile(r"```json\s*(\{.*?\})\s*```", _re.DOTALL | _re.IGNORECASE)
    for m in fence_pat.finditer(text):
        blocks.append(m.group(1))
    candidates = blocks or [text]

    def _first_brace_block(s: str) -> str | None:
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        in_squote = False
        in_dquote = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "'" and not in_dquote:
                in_squote = not in_squote
                continue
            if ch == '"' and not in_squote:
                in_dquote = not in_dquote
                continue
            if in_squote or in_dquote:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return None
    for c in candidates:
        try:
            data = _json.loads(c)
        except Exception:
            # Fallback: some models return Python-literal dicts with single quotes
            try:
                import ast as _ast
                data = _ast.literal_eval(c)
            except Exception:
                # Try extracting the first balanced-brace block from the text and parse that
                blk = _first_brace_block(c)
                if not blk:
                    blk = _first_brace_block(text)
                if not blk:
                    continue
                try:
                    data = _json.loads(blk)
                except Exception:
                    try:
                        import ast as _ast2
                        data = _ast2.literal_eval(blk)
                    except Exception:
                        continue
        if isinstance(data, dict):
            # Accept multiple possible keys for goals
            raw_goals = (
                data.get("goal_outputs")
                or data.get("goals")
                or data.get("goal")
            )
            if isinstance(raw_goals, str):
                raw_goals = [raw_goals]
            goals_list: list[str] = []
            if isinstance(raw_goals, list):
                for item in raw_goals:
                    if isinstance(item, str):
                        goals_list.append(item)
                    elif isinstance(item, dict):
                        # Handle objects like {"path template": "..."} or {"path_template": "..."}
                        # Normalize keys to be more permissive
                        if item:
                            # Build a normalized-key view (lowercased, collapsed whitespace, underscores -> spaces)
                            norm_map: dict[str, str] = {}
                            for k, v in item.items():
                                nk = str(k).lower().replace("_", " ")
                                nk = " ".join(nk.split())  # collapse whitespace
                                norm_map[nk] = v
                            candidates_keys = [
                                "path template", "path", "template", "output", "value",
                            ]
                            chosen: str | None = None
                            for k in candidates_keys:
                                if k in norm_map and isinstance(norm_map[k], str):
                                    chosen = norm_map[k]
                                    break
                            if chosen is None:
                                for v in item.values():
                                    if isinstance(v, str):
                                        chosen = v
                                        break
                            if chosen:
                                goals_list.append(chosen)
            # Known wildcards
            raw_known = data.get("known") or {}
            if isinstance(goals_list, list) and isinstance(raw_known, dict) and goals_list:
                # Dedup while preserving order
                seen = set()
                goals_norm = []
                for g in goals_list:
                    gs = str(g).strip()
                    if gs and gs not in seen:
                        seen.add(gs)
                        goals_norm.append(gs)
                # Basic sanity filter: drop obviously non-path placeholder words
                bad = {"path", "output", "template", "value"}
                goals_norm = [g for g in goals_norm if g.lower() not in bad]
                # If everything got filtered out, try regex fallback on the original text
                if not goals_norm:
                    import re as __re
                    # Match common path-like targets with config and/or wildcards
                    patt = __re.compile(r"\{config\.[^}]+\}[^\s'\"]+|[A-Za-z0-9._/-]*\{sample\}[A-Za-z0-9._/-]*")
                    hits = []
                    for m in patt.finditer(text):
                        val = m.group(0)
                        if val and val not in hits:
                            hits.append(val)
                    if hits:
                        goals_norm = hits
                known_norm = {str(k): str(v) for k, v in raw_known.items()}
                return goals_norm, known_norm
    return [], {}


def _filter_valid_goals(registry_yaml: str, goals: list[str]) -> tuple[list[str], list[str]]:
    """Keep only goals that some tool produces; return (kept, dropped)."""
    try:
        from pipemind.registry.schema import Registry
        import yaml as _yaml
        with open(registry_yaml, 'r', encoding='utf-8') as _f:
            reg = Registry.model_validate(_yaml.safe_load(_f))
        from pipemind.dag.builder import _match_output as _m_match  # type: ignore
    except Exception:
        return goals, []  # if anything goes wrong, don't block
    kept: list[str] = []
    dropped: list[str] = []
    tools = list(reg.tools.values())
    available_iotypes = {o.io_type for t in tools for o in t.outputs if o.io_type}
    for g in goals:
        if any(_m_match(t, g) for t in tools):
            kept.append(g)
        else:
            # Accept io_type prefixes like 'vcf:' (and ignore any suffix after ':')
            if ":" in g:
                prefix = g.split(":", 1)[0]
                if prefix in available_iotypes:
                    kept.append(g)
                    continue
            dropped.append(g)
    return kept, dropped


def _normalize_goal_templates(goals: list[str], known: dict[str, str]) -> list[str]:
    """Fix common model output issues:
    - Convert 'config.name/...' -> '{config.name}/...'
    - Replace known values back to wildcard placeholders: e.g., '.../S2...' -> '.../{sample}...'
    """
    out: list[str] = []
    for g in goals:
        s = g
        # restore {config.var}
        s = _re.sub(r"(?<!\{)(config\.[A-Za-z_][A-Za-z0-9_]*)/", r"{\1}/", s)
        # replace known wildcard values with their placeholders
        for k, v in (known or {}).items():
            if isinstance(v, str) and v:
                s = s.replace(v, "{" + k + "}")
        out.append(s)
    # de-dup while preserving order
    seen = set()
    norm = []
    for s in out:
        if s not in seen:
            seen.add(s)
            norm.append(s)
    return norm


@app.command()
def agent(
    registry_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Registry YAML"),
    max_turns: int = typer.Option(6, help="Max dialogue turns to obtain a valid plan"),
    model: str | None = typer.Option(None, help="Model id (Ollama/OpenAI)"),
    provider: str | None = typer.Option(None, help="LLM provider override"),
    run: bool = typer.Option(False, help="Execute after generation (not just dry-run)"),
    dry_run: bool = typer.Option(True, help="Perform Snakemake dry-run instead of full execution"),
    cores: int = typer.Option(4, help="Cores for Snakemake"),
):
    """Conversational agent: describe your desired final artifact; the model proposes goal outputs & wildcards.

    Workflow:
      1. User provides natural language goal.
      2. Model is instructed to emit JSON with goal_outputs & known wildcards.
      3. If JSON not found, user is prompted again.
      4. On success, a dynamic Snakefile is generated and (optionally) executed.
    """
    # Fancy welcome banner (printed before any LLM call)
    _banner = r"""

    ██████╗ ██╗██████╗ ███████╗███╗   ███╗██╗███╗   ██╗██████╗ 
    ██╔══██╗██║██╔══██╗██╔════╝████╗ ████║██║████╗  ██║██╔══██╗
    ██████╔╝██║██████╔╝█████╗  ██╔████╔██║██║██╔██╗ ██║██║  ██║
    ██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║╚██╔╝██║██║██║██║╚██╗██║  ██║
    ██║     ██║██║     ███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
    ╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ 

       PipeMind Agent · LLM-driven dynamic pipeline builder
    """
    print(f"[bold magenta]{_banner}[/bold magenta]")
    if provider or model:
        print(f"[dim]Session: provider={provider or 'auto'} model={model or 'auto'}[/dim]")

    # Load registry once to guide the model with allowed io_types
    try:
        from pipemind.registry.schema import Registry
        import yaml as _yaml
        with open(registry_yaml, 'r', encoding='utf-8') as __f:
            __reg = Registry.model_validate(_yaml.safe_load(__f))
        __io_types = sorted({o.io_type for t in __reg.tools.values() for o in t.outputs if o.io_type and o.io_type != 'unknown'})
    except Exception:
        __io_types = []

    io_hint = f" Allowed io_type prefixes: {', '.join(__io_types)}." if __io_types else ""
    intro = (
        "You are a pipeline planning assistant. Return JSON only.\n"
        "Schema: {\n  'goal_outputs': [<path template(s) or io_type:>],\n  'known': {<wildcard>: <value>}\n}\n"
        "Notes: 'io_type:' selects the last rule producing that type (e.g., 'vcf:')."
        + io_hint + " Use '{config.outdir}' and wildcards like '{sample}' in path templates when applicable."
    )
    history: list[dict[str, str]] = [
        {"role": "system", "content": intro},
    ]
    print("[bold cyan]Pipeline Agent started. Describe your desired final output (e.g., 'a filtered VCF for sample S1').[/bold cyan]")
    current_turn = 1
    history_turn = 0
    while current_turn <= max_turns:
        user_msg = typer.prompt(f"Turn {current_turn} - your instruction")
        history.append({"role": "user", "content": user_msg})
        try:
            reply_raw = llm_chat(history, model=model, provider=provider, json_mode=True)
            reply = reply_raw[0] if isinstance(reply_raw, tuple) else reply_raw
        except Exception as e:
            typer.secho(f"LLM error: {e}", fg="red")
            current_turn += 1
            continue
        history.append({"role": "assistant", "content": reply})
        g, k = _extract_plan(reply)
        if not k:
            try:
                k = _json.loads(user_msg).get("known", {}) if user_msg.strip().startswith("{") else {}
            except Exception:
                m = _re.search(r"\{\"?sample\"?\s*:\s*\"?([A-Za-z0-9_.-]+)\"?\}", user_msg)
                if m:
                    k = {"sample": m.group(1)}
        if g:
            g = _normalize_goal_templates(g, k)
            g_valid, g_dropped = _filter_valid_goals(registry_yaml, g)
            if g_dropped:
                typer.secho(f"Dropping {len(g_dropped)} unsupported goal(s): {g_dropped}", fg="yellow")
                if not g_valid:
                    more = (" Use one of the allowed io_type prefixes (e.g., 'vcf:') or a valid path template "
                            "from this registry using wildcards like '{sample}' and '{config.outdir}'.")
                    if __io_types:
                        more += f" Allowed io_types: {', '.join(__io_types)}."
                    typer.secho("Your JSON was valid, but no goals matched this registry." + more, fg="yellow")
            g = g_valid
        if not g:
            typer.secho("No acceptable plan yet. Provide a JSON object with goal_outputs and known.", fg="yellow")
            typer.echo(reply)
            current_turn += 1
            continue
        # Have a plan: try to generate & run
        goals, known = g, k
        typer.secho(f"Extracted plan: goals={goals} known={known}", fg="green")
        try:
            result = materialize_and_optionally_run(
                registry_yaml=registry_yaml,
                goal_outputs=goals,
                known=known,
                workdir=".pipemind_runs/agent_last",
                run=True,
                dry_run=dry_run,
                cores=cores,
            )
        except Exception as e:
            typer.secho(f"Generation/Execution error: {e}", fg="red")
            # Feed error back to the model and continue
            history.append({"role": "system", "content": f"Execution error occurred: {e}. Please propose a revised JSON plan."})
            current_turn += 1
            continue
        typer.secho(f"Snakefile: {result['snakefile']}", fg="blue")
        if result.get("stdout"):
            typer.echo(result["stdout"][-800:])
        # If success (or dry-run success), finish; else keep the conversation going
        if result.get("returncode", 0) == 0:
            typer.secho("Agent run complete.", fg="green")
            return
        # Failure path: provide a compact summary to the LLM and user, then continue
        typer.secho("Execution failed. Returning to planning loop.", fg="red")
        err_tail = (result.get("stderr") or "")[-400:]
        out_tail = (result.get("stdout") or "")[-400:]
        summary = (
            "Snakemake execution failed. Here is a brief summary of the outputs. "
            "Revise the plan to choose a different final artifact (possibly an earlier step) or adjust wildcards.\n"
            f"STDOUT tail:\n{out_tail}\nSTDERR tail:\n{err_tail}"
        )
        history.append({"role": "system", "content": summary})
        current_turn += 1
    # If we exit the loop without success
    typer.secho("No successful execution within the turn limit.", fg="red")
    raise typer.Exit(code=1)


@app.command()
def compose(
    registry_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Registry YAML"),
    goal: str | None = typer.Argument(None, help="(Deprecated) Single goal output template or io_type: prefix"),
    outputs: list[str] = typer.Option(
        [], "-o", "--output", help="One or more goal output templates or io_type: prefixes (legacy compat)"
    ),
    known: str | None = typer.Option(None, help='JSON string of known wildcards (e.g. {"sample":"S1"})'),
    fill: str | None = typer.Option(
        None,
        help="Alias of --known (legacy) JSON mapping of wildcard -> value to concretise rule all",
    ),
    workdir: str | None = typer.Option(None, help="Work directory for generated Snakefile"),
    run: bool = typer.Option(False, help="Execute Snakemake after generation"),
    dry_run: bool = typer.Option(True, help="Dry-run Snakemake (-n)"),
    cores: int = typer.Option(4, help="Cores for Snakemake"),
):
    """Generate (and optionally execute) a minimal dynamic Snakefile with a robust rule all.

    Backward compatible modes:
      New: pipemind compose -o results/{sample}.final.txt --known '{"sample":"S1"}' --run --dry-run
      Old: pipemind compose -o results/{sample}.final.txt --fill '{"sample":"S1"}' --run --dry-run
      Old (single positional): pipemind compose 'results/{sample}.final.txt' --known '{"sample":"S1"}'
    """
    # Merge goal/outputs logic
    goal_outputs: list[str] = []
    if outputs:
        goal_outputs.extend(outputs)
    if goal:
        if goal_outputs:
            typer.secho("Both positional goal and --output provided; using all combined.", fg="yellow")
        goal_outputs.append(goal)
    if not goal_outputs:
        raise typer.BadParameter("Provide at least one goal via positional argument or -o/--output")

    # Consolidate known / fill
    wildcard_json = known or fill
    try:
        known_dict = _json.loads(wildcard_json) if wildcard_json else {}
    except Exception as e:  # pragma: no cover - defensive
        raise typer.BadParameter(f"Invalid JSON for wildcards (--known/--fill): {e}")

    # Run generation (and optional execution)
    try:
        res = materialize_and_optionally_run(
            registry_yaml=registry_yaml,
            goal_outputs=goal_outputs,
            known=known_dict,
            workdir=workdir,
            run=run,
            dry_run=dry_run,
            cores=cores,
        )
    except Exception as e:
        typer.secho(f"Error: {e}", fg="red")
        raise typer.Exit(code=1)
    typer.echo(f"Snakefile: {res['snakefile']}")
    if res.get("stdout"):
        typer.echo(res["stdout"][-1200:])
    if run and not dry_run and res.get("returncode", 1) != 0:
        raise typer.Exit(code=res.get("returncode", 1))


## (Removed duplicate compose command to avoid name collision)


if __name__ == "__main__":
    app()
