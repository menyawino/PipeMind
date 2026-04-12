from __future__ import annotations
from typing import Dict, List, Tuple, Any
import yaml
import re

from pipemind.registry.schema import Registry, ToolSpec


def _load_registry(registry_yaml: str) -> Registry:
    with open(registry_yaml, 'r') as f:
        return Registry.model_validate(yaml.safe_load(f))


WILDCARD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
SYMBOLIC_REF_RE = re.compile(r"^rules\.([a-zA-Z_][a-zA-Z0-9_]*)\.output\.([a-zA-Z_][a-zA-Z0-9_]*)$")


def _template_to_regex(path_template: str) -> re.Pattern[str]:
    """Convert a path template into a concrete-path matcher.

    - Wildcards like {sample} match one path segment.
    - Config placeholders like {config.outdir} are treated as concrete text,
      because they should typically be resolved before matching concrete targets.
    """
    pattern = []
    pos = 0
    for m in WILDCARD_RE.finditer(path_template):
        pattern.append(re.escape(path_template[pos:m.start()]))
        wildcard_name = m.group(1)
        if wildcard_name.startswith("config."):
            pattern.append(re.escape("{" + wildcard_name + "}"))
        else:
            pattern.append(r"[^/]+")
        pos = m.end()
    pattern.append(re.escape(path_template[pos:]))
    return re.compile(r"^" + "".join(pattern) + r"$")


def _canonical_template(path: str) -> str:
    """Normalize template structure by replacing wildcard names with {}."""
    return WILDCARD_RE.sub("{}", path)


def _match_output(tool: ToolSpec, goal: str) -> bool:
    """Return True if a tool has an output template that matches the goal."""
    for o in tool.outputs:
        if not o.path_template:
            continue
        t = o.path_template
        if goal == t:
            return True
        if _template_to_regex(t).match(goal):
            return True
    return False


def _collect_wildcards(tool: ToolSpec) -> List[str]:
    wc = set()
    for i in tool.inputs + tool.outputs:
        if i.path_template:
            wc.update(WILDCARD_RE.findall(i.path_template))
    return sorted(wc)


def _outputs_compatible(inp_io: str, out_io: str) -> bool:
    if inp_io == "unknown" or out_io == "unknown":
        return True
    return inp_io == out_io


def _find_producers_for_input(inp, tools: List[ToolSpec], tools_by_rule: Dict[str, ToolSpec], current_tool_id: str) -> List[ToolSpec]:
    if not inp.path_template:
        return []

    # 1) Explicit symbolic references are authoritative.
    m = SYMBOLIC_REF_RE.match(inp.path_template)
    if m:
        rule_name, out_name = m.groups()
        candidate = tools_by_rule.get(rule_name)
        if candidate and candidate.id != current_tool_id:
            for o in candidate.outputs:
                if o.name == out_name:
                    return [candidate]
        return []

    # 2) Prefer structural path-template equivalence over loose io_type matching.
    inp_canon = _canonical_template(inp.path_template)
    path_candidates: List[ToolSpec] = []
    for t in tools:
        if t.id == current_tool_id:
            continue
        for o in t.outputs:
            if not o.path_template:
                continue
            if _canonical_template(o.path_template) == inp_canon and _outputs_compatible(inp.io_type, o.io_type):
                path_candidates.append(t)
                break
    if path_candidates:
        return path_candidates

    # 3) Conservative fallback: only allow io_type linkage when uniquely determined.
    if inp.io_type == "unknown":
        return []
    io_candidates = [
        t for t in tools
        if t.id != current_tool_id and any(o.io_type == inp.io_type for o in t.outputs)
    ]
    return io_candidates


def _filter_terminals_by_suffix(terminals: List[ToolSpec], suffix: str) -> List[ToolSpec]:
    if not suffix:
        return terminals
    filtered: List[ToolSpec] = []
    suffix_canon = _canonical_template(suffix)
    for t in terminals:
        if _match_output(t, suffix):
            filtered.append(t)
            continue
        for o in t.outputs:
            if o.path_template and _canonical_template(o.path_template) == suffix_canon:
                filtered.append(t)
                break
    return filtered


def build_dag_for_goal(registry_yaml: str, goal_output: str, known: Dict[str, Any]) -> Dict[str, Any]:
    reg = _load_registry(registry_yaml)
    tools = list(reg.tools.values())
    tools_by_rule = {t.rule: t for t in tools}

    # Identify terminal tool(s) that produce goal_output
    terminal = [t for t in tools if _match_output(t, goal_output)]
    if not terminal:
        # fallback: io type prefix (e.g. vcf:) is only accepted when unambiguous
        if ":" in goal_output:
            iotype, suffix = goal_output.split(":", 1)
            terminal = [t for t in tools if any(o.io_type == iotype for o in t.outputs)]
            terminal = _filter_terminals_by_suffix(terminal, suffix)
    if not terminal:
        raise ValueError(f"No tool produces goal: {goal_output}")
    if len(terminal) > 1:
        raise ValueError(
            f"Ambiguous goal '{goal_output}' produced by multiple tools: "
            + ", ".join(sorted(t.id for t in terminal))
            + ". Provide a concrete goal output path to disambiguate."
        )

    plan_steps: List[Dict[str, Any]] = []
    visited = set()
    in_progress = set()

    def backchain(tool: ToolSpec):
        if tool.id in visited:
            return
        if tool.id in in_progress:
            return
        in_progress.add(tool.id)
        # Recurse on prerequisites by explicit provenance first; reject ambiguity.
        for inp in tool.inputs:
            producers = _find_producers_for_input(inp, tools, tools_by_rule, tool.id)
            if len(producers) > 1:
                raise ValueError(
                    f"Ambiguous producer resolution for input '{inp.name}' of tool '{tool.id}'. "
                    + "Candidates: "
                    + ", ".join(sorted(t.id for t in producers))
                )
            if len(producers) == 1:
                backchain(producers[0])
        in_progress.discard(tool.id)
        visited.add(tool.id)
        plan_steps.append({
            "tool": tool.id,
            "wildcards": {w: known.get(w) for w in _collect_wildcards(tool)},
        })

    backchain(terminal[0])

    return {
        "goal": goal_output,
        "steps": plan_steps,
        "missing": sorted({w for step in plan_steps for w,v in step["wildcards"].items() if v is None}),
    }
