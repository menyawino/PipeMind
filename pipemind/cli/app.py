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


@app.command()
def compose(
    registry_yaml: str = typer.Option("pipemind/registry/registry.yaml", help="Registry YAML path"),
    outputs: list[str] = typer.Option(..., '--output', '-o', help="One or more desired final output path templates (wildcards allowed)"),
    fill: str | None = typer.Option(None, help="JSON mapping of wildcard -> value to concretise rule all"),
    run: bool = typer.Option(False, help="Run snakemake after generating the dynamic Snakefile"),
    dry_run: bool = typer.Option(False, help="If running, perform Snakemake dry-run (-n) only"),
    workdir: str | None = typer.Option(None, help="Directory to write the dynamic Snakefile (default .pipemind_runs/<timestamp>)"),
    cores: int = typer.Option(4, help="Cores for Snakemake execution"),
):
    """Generate (and optionally execute) a minimal dynamic Snakefile with a robust rule all.

    Example:
      pipemind compose -o results/{sample}.vcf.gz --fill '{"sample":"NA12878"}' --run
    """
    import json as _json
    known = _json.loads(fill) if fill else {}
    try:
        result = materialize_and_optionally_run(
            registry_yaml=registry_yaml,
            goal_outputs=outputs,
            known=known,
            workdir=workdir,
            run=run,
            dry_run=dry_run,
            cores=cores,
        )
    except Exception as e:  # pragma: no cover - CLI integration path
        print(f"[red]Compose failed:[/red] {e}")
        raise typer.Exit(code=1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
