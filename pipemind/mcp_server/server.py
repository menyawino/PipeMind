from __future__ import annotations
from typing import Dict, Any, Callable
import os
import json
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.tools import Tool
from pydantic import BaseModel

from pipemind.registry.schema import Registry, ToolSpec
from pipemind.utils.audit import write_invocation_log


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

    def _call(**kwargs):
        # Expect 'target' key: which output to build; default to first output
        outputs = tool.outputs
        if not outputs:
            return {"status": "no-outputs"}
        target_template = outputs[0].path_template or ""
        # Basic wildcard substitution from kwargs
        target = target_template
        for k, v in kwargs.items():
            target = target.replace(f"{{{k}}}", str(v))

        workdir = os.getcwd()
        # Invoke snakemake to build the target
        snakefile = os.getenv("PIPEMIND_SNAKEFILE", "WES-Pipeline-Snakemake/workflow/Snakefile")
        cmd = [
            "snakemake",
            "-s", snakefile,
            target,
            "-c", "1",
            "--rerun-incomplete",
        ]
        res = run(cmd, capture_output=True, text=True, cwd=workdir)
        payload = {
            "returncode": res.returncode,
            "stdout": res.stdout[-4000:],
            "stderr": res.stderr[-4000:],
            "target": target,
            "tool": tool.id,
            "kwargs": kwargs,
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
            else:
                t = "string"
            props[p.name] = {"type": t}
            if p.required:
                required.append(p.name)

        mcp.add_tool(
            CallableTool(
                name=tool.id,
                description=tool.description or tool.name,
                parameters={
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
                fn=make_tool_callable(tool),
            )
        )

    api = FastAPI(title="PipeMind MCP Server")

    @api.get("/schemas", response_model=SchemasResponse)
    def schemas():
        return SchemasResponse(registry=reg.model_dump())

    # Mount MCP server under /mcp if needed by hosting environment
    # Note: FastMCP typically runs over stdio; here we expose http for schema discovery
    return api, mcp
