from __future__ import annotations
from typing import Dict, Any, List
import yaml
import re


def _load_registry(path: str):
    from pipemind.registry.schema import Registry
    with open(path, 'r') as f:
        return Registry.model_validate(yaml.safe_load(f))


def required_fields_for_goal(registry_yaml: str, goal: str) -> List[str]:
    reg = _load_registry(registry_yaml)
    import os
    def match_output(tool):
        for o in tool.outputs:
            if o.path_template and (goal == o.path_template or goal.endswith(os.path.basename(o.path_template))):
                return True
        return False
    target = next((t for t in reg.tools.values() if match_output(t)), None)
    if not target:
        return []
    req = set()
    for io in target.inputs + target.outputs:
        if io.path_template:
            req.update(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", io.path_template))
    req.update(p.name for p in target.params if p.required)
    return sorted(req)


def interactive_collect(fields: List[str]) -> Dict[str, Any]:
    out = {}
    for f in fields:
        val = input(f"Enter value for {f}: ")
        out[f] = val
    return out
