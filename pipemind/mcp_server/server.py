from __future__ import annotations
from typing import Dict, Any, Callable
import os
import json
import platform
import sys
from datetime import datetime, timezone
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.tools import Tool
from pydantic import BaseModel

from pipemind.registry.schema import Registry, ToolSpec
from pipemind.tools.runner import resolve_snakemake_command
from pipemind.utils.audit import write_invocation_log, file_sha256
from pipemind.snakemake.generator import materialize_and_optionally_run


def load_registry(registry_path: str) -> Registry:
    import yaml
    with open(registry_path, 'r') as f:
        data = yaml.safe_load(f)
    return Registry.model_validate(data)


def make_tool_callable(tool: ToolSpec) -> Callable[..., Any]:
    """Return a callable that executes the underlying Snakemake rule via snakemake target outputs.

    For prototype: we call Snakemake with the first output path to build.
    """
    from subprocess import run

    def _resolve_template(path_template: str, values: Dict[str, Any]) -> str:
        resolved = path_template
        for k, v in values.items():
            if isinstance(v, (str, int, float, bool)):
                resolved = resolved.replace(f"{{{k}}}", str(v))
        return resolved

    def _call(**kwargs):
        outputs = [o for o in tool.outputs if o.path_template]
        if not outputs:
            return {"status": "no-outputs"}

        explicit_target = kwargs.get("target")
        requested_output_name = kwargs.get("output_name")

        if explicit_target:
            target = str(explicit_target)
        else:
            selected = None
            if requested_output_name:
                selected = next((o for o in outputs if o.name == requested_output_name), None)
                if selected is None:
                    return {
                        "status": "error",
                        "error": f"Unknown output_name '{requested_output_name}'. Available: {[o.name for o in outputs]}",
                    }
            elif len(outputs) == 1:
                selected = outputs[0]
            else:
                return {
                    "status": "error",
                    "error": "Multiple outputs available; provide output_name or target to disambiguate.",
                    "available_outputs": [o.name for o in outputs],
                }
            target = _resolve_template(selected.path_template or "", kwargs)

        workdir = os.getcwd()
        # Invoke snakemake to build the target
        snakefile = os.getenv("PIPEMIND_SNAKEFILE", "WES-Pipeline-Snakemake/workflow/Snakefile")
        cmd = [
            *resolve_snakemake_command(),
            "-s", snakefile,
            target,
            "-c", "1",
            "--rerun-incomplete",
        ]
        res = run(cmd, capture_output=True, text=True, cwd=workdir)

        resolved_outputs = {
            o.name: _resolve_template(o.path_template or "", kwargs)
            for o in outputs
        }
        output_hashes = {
            name: file_sha256(path)
            for name, path in resolved_outputs.items()
            if path and os.path.exists(path)
        }

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "stdout_bytes": len(res.stdout.encode("utf-8", errors="ignore")),
            "stderr_bytes": len(res.stderr.encode("utf-8", errors="ignore")),
            "target": target,
            "resolved_outputs": resolved_outputs,
            "output_hashes_sha256": output_hashes,
            "tool": tool.id,
            "kwargs": kwargs,
            "cmd": cmd,
            "environment": {
                "python": sys.version.split()[0],
                "platform_system": platform.system(),
                "platform_release": platform.release(),
                "platform_machine": platform.machine(),
                "cwd": workdir,
                "snakefile": snakefile,
            },
        }
        write_invocation_log(os.path.join(".pipemind", "audit"), payload)
        return payload

    return _call


class SchemasResponse(BaseModel):
    registry: Dict[str, Any]


def create_app(registry_path: str) -> tuple[FastAPI, FastMCP]:
    reg = load_registry(registry_path)
    mcp = FastMCP("pipemind-mcp")

    # Local Tool subclass to bridge our callable into FastMCP's tool interface
    class CallableTool(Tool):
        fn: Callable[..., Any]

        async def run(self, arguments: Dict[str, Any]):  # type: ignore[override]
            res = self.fn(**(arguments or {}))
            if hasattr(res, "__await__"):
                res = await res  # type: ignore[func-returns-value]
            # Let FastMCP serialize result into content
            from fastmcp.tools.tool import ToolResult

            return ToolResult(content=res)

    # Register tools dynamically
    for tool in reg.tools.values():
        # Build a JSON schema for inputs based on wildcards/params
        props: Dict[str, Any] = {}
        required = []
        # Collect wildcards from outputs and inputs templates
        import re
        wildcards = set()
        for i in tool.inputs + tool.outputs:
            if i.path_template:
                wildcards.update(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", i.path_template))
        for w in sorted(wildcards):
            props[w] = {"type": "string"}
            required.append(w)
        for p in tool.params:
            if p.param_type == "int":
                t = "integer"
            elif p.param_type == "float":
                t = "number"
            elif p.param_type == "bool":
                t = "boolean"
            elif p.param_type == "json":
                t = ["object", "array", "string", "number", "integer", "boolean"]
            else:
                t = "string"
            description = p.description
            if p.binding_kind and p.binding_target:
                binding_note = f"Binds dynamic {p.binding_kind} '{p.binding_target}'."
                description = f"{description} {binding_note}" if description else binding_note
            props[p.name] = {"type": t}
            if description:
                props[p.name]["description"] = description
            if p.required:
                required.append(p.name)

        output_names = [o.name for o in tool.outputs if o.path_template]
        if output_names:
            props["output_name"] = {
                "type": "string",
                "enum": output_names,
                "description": "Optional output selector for multi-output rules.",
            }
        props["target"] = {
            "type": "string",
            "description": "Optional explicit concrete target path. Overrides output_name/wildcard expansion.",
        }

        description = tool.description or tool.name
        notes = tool.agent_ready_notes or tool.composition_ready_notes
        if notes:
            description = description + " [agent notes: " + "; ".join(notes) + "]"

        mcp.add_tool(
            CallableTool(
                name=tool.id,
                description=description,
                parameters={
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
                fn=make_tool_callable(tool),
            )
        )

    # Add a high-level composition tool that uses the dynamic generator
    def _compose_fn(**kwargs):
        outputs = kwargs.get("outputs") or kwargs.get("goals") or []
        if isinstance(outputs, str):
            outputs = [outputs]
        known = kwargs.get("known") or {}
        run = bool(kwargs.get("run", False))
        dry_run = bool(kwargs.get("dry_run", True))
        workdir = kwargs.get("workdir")
        cores = int(kwargs.get("cores", 4))
        res = materialize_and_optionally_run(
            registry_yaml=registry_path,
            goal_outputs=outputs,
            known=known,
            workdir=workdir,
            run=run,
            dry_run=dry_run,
            cores=cores,
        )
        snakefile_path = str(res.get("snakefile", ""))
        snakefile_hash = file_sha256(snakefile_path) if snakefile_path and os.path.exists(snakefile_path) else None
        write_invocation_log(os.path.join(".pipemind", "audit"), {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool": "pipemind.compose",
            "kwargs": kwargs,
            "goal_outputs": outputs,
            "known": known,
            "result": {k: v for k, v in res.items() if k in ("snakefile", "workdir", "returncode")},
            "snakefile_sha256": snakefile_hash,
        })
        return res

    mcp.add_tool(
        CallableTool(
            name="pipemind.compose",
            description=(
                "Generate a minimal Snakefile for the requested goal outputs and optionally run Snakemake. "
                "Accepts {outputs: string[]|string, known: object, run?: boolean, dry_run?: boolean, workdir?: string, cores?: integer}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "outputs": {"type": ["array", "string"], "description": "Goal outputs (templates allowed)"},
                    "known": {"type": "object", "description": "Wildcard/param substitutions"},
                    "run": {"type": "boolean"},
                    "dry_run": {"type": "boolean"},
                    "workdir": {"type": "string"},
                    "cores": {"type": "integer"},
                },
                "required": ["outputs"],
            },
            fn=_compose_fn,
        )
    )

    api = FastAPI(title="PipeMind MCP Server")

    @api.get("/schemas", response_model=SchemasResponse)
    def schemas():
        return SchemasResponse(registry=reg.model_dump())

    # Mount MCP server under /mcp if needed by hosting environment
    # Note: FastMCP typically runs over stdio; here we expose http for schema discovery
    return api, mcp
