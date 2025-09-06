from __future__ import annotations
from typing import Dict, List, Tuple, Any
import os
import yaml
import re

from pipemind.registry.schema import Registry, ToolSpec


def _load_registry(registry_yaml: str) -> Registry:
    with open(registry_yaml, 'r') as f:
        return Registry.model_validate(yaml.safe_load(f))


def _match_output(tool: ToolSpec, goal: str) -> bool:
    # Match if goal equals or endswith the path template stripped of config vars
    for o in tool.outputs:
        if not o.path_template:
            continue
        t = o.path_template
        if goal == t:
            return True
        if goal.endswith(os.path.basename(t)):
            return True
    return False


def _collect_wildcards(tool: ToolSpec) -> List[str]:
    wc = set()
    for i in tool.inputs + tool.outputs:
        if i.path_template:
            wc.update(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", i.path_template))
    return sorted(wc)


def build_dag_for_goal(registry_yaml: str, goal_output: str, known: Dict[str, Any]) -> Dict[str, Any]:
    reg = _load_registry(registry_yaml)
    tools = list(reg.tools.values())

    # Identify terminal tool(s) that produce goal_output
    terminal = [t for t in tools if _match_output(t, goal_output)]
    if not terminal:
        # fallback: try io_type preface like vcf: to pick last tool producing that type
        if ":" in goal_output:
            iotype = goal_output.split(":", 1)[0]
            terminal = [t for t in tools if any(o.io_type == iotype for o in t.outputs)]
    if not terminal:
        raise ValueError(f"No tool produces goal: {goal_output}")

    plan_steps: List[Dict[str, Any]] = []
    visited = set()

    def backchain(tool: ToolSpec):
        if tool.id in visited:
            return
        # Recurse on prerequisites by linking each input to a producer if possible
        for inp in tool.inputs:
            prod = None
            for t in tools:
                if any(o.io_type == inp.io_type for o in t.outputs):
                    prod = t
                    break
            if prod and prod.id != tool.id:
                backchain(prod)
        visited.add(tool.id)
        plan_steps.append({
            "tool": tool.id,
            "wildcards": {w: known.get(w) for w in _collect_wildcards(tool) if w in known},
        })

    backchain(terminal[0])

    return {
        "goal": goal_output,
        "steps": plan_steps,
        "missing": sorted({w for step in plan_steps for w,v in step["wildcards"].items() if v is None}),
    }
