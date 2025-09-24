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
    for c in candidates:
        try:
            data = _json.loads(c)
        except Exception:
            continue
        if isinstance(data, dict) and "goal_outputs" in data:
            goals = data.get("goal_outputs") or []
            known = data.get("known") or {}
            if isinstance(goals, list) and isinstance(known, dict):
                # normalize to str keys/values for known
                known_norm = {str(k): str(v) for k, v in known.items()}
                goals_norm = [str(g) for g in goals]
                return goals_norm, known_norm
    return [], {}


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
    intro = (
        "You are a pipeline planning assistant. Given the user's bioinformatics aim, respond with JSON only.\n"
        "JSON schema: {\n  'goal_outputs': [<path template(s) or io_type: prefixed spec>],\n  'known': {<wildcard>: <value>}\n}\n"
        "List only outputs that the workflow should produce. Provide wildcards you can confidently fill; omit unknowns."
    )
    history: list[dict[str, str]] = [
        {"role": "system", "content": intro},
    ]
    typer.echo("[bold cyan]Pipeline Agent started. Describe your desired final output (e.g., 'a filtered VCF for sample S1').[/bold cyan]")
    goals: list[str] = []
    known: dict[str, str] = {}
    for turn in range(1, max_turns + 1):
        user_msg = typer.prompt(f"Turn {turn} - your instruction")
        history.append({"role": "user", "content": user_msg})
        try:
            reply_raw = llm_chat(history, model=model, provider=provider)
            reply = reply_raw[0] if isinstance(reply_raw, tuple) else reply_raw  # normalize to str
        except Exception as e:
            typer.secho(f"LLM error: {e}", fg="red")
            continue
        history.append({"role": "assistant", "content": reply})
        g, k = _extract_plan(reply)
        if g:
            goals, known = g, k
            typer.secho(f"Extracted plan: goals={goals} known={known}", fg="green")
            break
        else:
            typer.secho("No valid JSON plan detected. Please refine your description (model reply shown below):", fg="yellow")
            typer.echo(reply)
    if not goals:
        typer.secho("Failed to obtain a structured plan within turn limit.", fg="red")
        raise typer.Exit(code=1)
    # Generate & optionally run dynamic Snakefile
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
        raise typer.Exit(code=1)
    typer.secho(f"Snakefile: {result['snakefile']}", fg="blue")
    if result.get("stdout"):
        typer.echo(result["stdout"][-800:])
    if (not dry_run) and result.get("returncode", 1) != 0:
        typer.secho("Execution failed.", fg="red")
        raise typer.Exit(code=1)
    typer.secho("Agent run complete.", fg="green")


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
