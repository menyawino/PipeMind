from __future__ import annotations
"""Dynamic Snakemake workflow materialisation.

This module converts a subset of the registry (tools required to produce one or
more goal outputs) into a self-contained Snakefile containing:

rule all:
    input: <resolved goal targets>

Plus one rule per required ToolSpec reconstructed from its recorded metadata.

Reliability features:
 - Validation of missing wildcard values required to fully resolve `rule all`.
 - Stable ordering (topological-ish) based on back-chaining order used by
   `build_dag_for_goal` to reduce spurious diffs.
 - Defensive escaping of quotes in shell/paths.
 - Optional dry-run safe generation (no execution side effects).

Assumptions:
 - Path templates already Snakemake-compatible (registry.parser enforces).
 - Wildcards that remain unresolved in intermediate rules are acceptable; only
   the `rule all` targets must be concrete for an execution run.
"""

from dataclasses import dataclass
from typing import Dict, List, Iterable, Set, Any, Tuple
import os
import re
import time
import textwrap

from pipemind.registry.schema import ToolSpec, Registry
from pipemind.dag.builder import build_dag_for_goal
import yaml

WILDCARD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _load_registry(registry_yaml: str) -> Registry:
    with open(registry_yaml, "r", encoding="utf-8") as f:
        return Registry.model_validate(yaml.safe_load(f))


def _collect_required_tools(registry: Registry, goals: List[str], known: Dict[str, Any]) -> Tuple[List[ToolSpec], List[str]]:
    """Return ordered list of ToolSpec needed for goals and list of missing wildcards.

    We reuse build_dag_for_goal individually for each goal to recover ordering.
    Duplicates are removed preserving first-seen order.
    """
    ordered: List[ToolSpec] = []
    seen: Set[str] = set()
    missing_all: Set[str] = set()
    for g in goals:
        # Lightweight inline back-chain using builder helper functions without re-reading registry from disk.
        from pipemind.dag.builder import _match_output as _m_match, _collect_wildcards as _m_wc  # type: ignore
        tools = list(registry.tools.values())
        terminal = [t for t in tools if _m_match(t, g)]
        if not terminal and ":" in g:
            iotype = g.split(":", 1)[0]
            terminal = [t for t in tools if any(o.io_type == iotype for o in t.outputs)]
        if not terminal:
            raise ValueError(f"No tool produces goal: {g}")
        local_steps: List[ToolSpec] = []
        visited: Set[str] = set()

        def backchain(tool: ToolSpec):
            if tool.id in visited:
                return
            # Recurse over inputs by io_type producer matching (first match heuristic)
            for inp in tool.inputs:
                prod = None
                for t in tools:
                    if any(o.io_type == inp.io_type for o in t.outputs):
                        prod = t
                        break
                if prod and prod.id != tool.id:
                    backchain(prod)
            visited.add(tool.id)
            local_steps.append(tool)

        backchain(terminal[0])
        # Record missing wildcard values for these steps
        for step_tool in local_steps:
            for wc in _m_wc(step_tool):  # type: ignore
                if wc not in known:
                    missing_all.add(wc)
        # Merge into global ordered list
        for t in local_steps:
            if t.id not in seen:
                ordered.append(t)
                seen.add(t.id)
    return ordered, sorted(missing_all)


def _safe_substitute(template: str, mapping: Dict[str, Any]) -> str:
    def repl(m):
        key = m.group(1)
        if key in mapping and mapping[key] is not None:
            return str(mapping[key])
        return "{" + key + "}"  # leave unresolved
    return WILDCARD_RE.sub(repl, template)


def _format_kv_block(kind: str, items: List[Tuple[str, str]]) -> str:
    if not items:
        return ""
    # indent subsequent lines
    inner = ",\n        ".join([f"{k}='{v}'" for k, v in items])
    return f"    {kind}:\n        {inner}\n"


def _escape_shell(cmd: str) -> str:
    # Basic sanitisation; we rely on Snakemake quoting for more complex cases.
    return cmd.replace('"', '\\"')


def generate_snakefile(
    registry_yaml: str,
    goal_outputs: List[str],
    known: Dict[str, Any] | None = None,
    enforce_all_concrete: bool = True,
) -> str:
    """Return text of a Snakefile for the requested goals.

    If `enforce_all_concrete` is True, unresolved wildcards in final goal target
    paths raise an error (they would make rule all ambiguous).
    """
    known = known or {}
    registry = _load_registry(registry_yaml)
    tools, missing = _collect_required_tools(registry, goal_outputs, known)
    # Resolve concrete final targets
    concrete_targets: List[str] = []
    unresolved_final: List[str] = []
    for g in goal_outputs:
        tgt = _safe_substitute(g, known)
        if WILDCARD_RE.search(tgt):
            unresolved_final.append(tgt)
        concrete_targets.append(tgt)
    if enforce_all_concrete and unresolved_final:
        raise ValueError(
            "Unresolved wildcards in goal outputs: " + ", ".join(unresolved_final) +
            f". Provide values for: {', '.join(sorted({w for t in unresolved_final for w in WILDCARD_RE.findall(t)}))}"
        )

    header = textwrap.dedent(
        f"""# Auto-generated by pipemind.snakemake.generator at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"""
    )
    rule_all = "rule all:\n    input:\n        " + ",\n        ".join([f"'{t}'" for t in concrete_targets]) + "\n\n"

    rule_texts: List[str] = []
    for t in tools:
        # Build input/output param blocks
        in_items: List[Tuple[str, str]] = []
        for i, idecl in enumerate(t.inputs):
            nm = idecl.name if not idecl.name.startswith("_") else f"in{i+1}"
            if not idecl.path_template:
                continue
            in_items.append((nm, idecl.path_template))
        out_items: List[Tuple[str, str]] = []
        for i, odecl in enumerate(t.outputs):
            nm = odecl.name if not odecl.name.startswith("_") else f"out{i+1}"
            if not odecl.path_template:
                continue
            out_items.append((nm, odecl.path_template))
        param_items: List[Tuple[str, str]] = []
        for p in t.params:
            # Store description or empty placeholder; user can override via CLI config
            if p.default is not None:
                param_items.append((p.name, str(p.default)))
        # Compose rule body
        body_lines = []
        body_lines.append(f"rule {t.rule}:")
        if in_items:
            body_lines.append(_format_kv_block("input", in_items).rstrip())
        if out_items:
            body_lines.append(_format_kv_block("output", out_items).rstrip())
        if param_items:
            body_lines.append(_format_kv_block("params", param_items).rstrip())
        if t.threads:
            body_lines.append(f"    threads: {t.threads}")
        if t.conda_env:
            body_lines.append(f"    conda: '{t.conda_env}'")
        if t.wrapper:
            body_lines.append(f"    wrapper: '{t.wrapper}'")
        if t.container:
            body_lines.append(f"    container: '{t.container}'")
        if t.resources:
            res_kv = ", ".join([f"{k}={v!r}" for k, v in t.resources.items()])
            body_lines.append(f"    resources: {res_kv}")
        # Prefer original shell command
        if t.command:
            body_lines.append(f"    shell: \"{_escape_shell(t.command)}\"")
        elif t.script:
            body_lines.append(f"    script: '{t.script}'")
        elif t.run_code:
            # embed run block preserving indentation
            rc = textwrap.indent(t.run_code, "        ")
            body_lines.append("    run:")
            body_lines.append(rc)
        else:
            body_lines.append("    shell: 'echo " + t.rule + " executed'" )
        rule_texts.append("\n".join([ln for ln in body_lines if ln.strip() != ""]))

    return header + rule_all + "\n\n".join(rule_texts) + "\n"


def materialize_and_optionally_run(
    registry_yaml: str,
    goal_outputs: List[str],
    known: Dict[str, Any] | None = None,
    workdir: str | None = None,
    run: bool = False,
    dry_run: bool = False,
    cores: int = 4,
) -> Dict[str, Any]:
    """Generate a Snakefile and optionally execute snakemake.

    Returns metadata including path to Snakefile and (if run) execution details.
    """
    known = known or {}
    wd = workdir or os.path.join(".pipemind_runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(wd, exist_ok=True)
    snakefile_path = os.path.join(wd, "Snakefile")
    content = generate_snakefile(registry_yaml, goal_outputs, known)
    with open(snakefile_path, "w", encoding="utf-8") as f:
        f.write(content)
    result: Dict[str, Any] = {"snakefile": snakefile_path, "workdir": wd}
    if run:
        import subprocess, shutil
        snakemake_bin = shutil.which("snakemake")
        if not snakemake_bin:
            raise RuntimeError("Snakemake executable not found in PATH")
        cmd = [snakemake_bin, "-s", snakefile_path, "-c", str(cores)]
        if dry_run:
            cmd.append("-n")
        cmd.extend(["--rerun-incomplete", "--printshellcmds", "--quiet"])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result.update({
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "cmd": " ".join(cmd),
        })
    return result
