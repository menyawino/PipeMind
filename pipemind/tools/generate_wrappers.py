from __future__ import annotations
from typing import Dict, Any
import os
import yaml
from pipemind.registry.schema import Registry


TEMPLATE = '''from __future__ import annotations
from typing import Any, Dict
from pipemind.tools.runner import run_snakemake_target


def {func_name}(**kwargs) -> Dict[str, Any]:
    """Auto-generated wrapper for Snakemake rule {rule}.

    Parameters are substituted into the output template to produce a concrete target path.
    """
    target = "{target_template}"
    for k, v in kwargs.items():
        target = target.replace("{" + str(k) + "}", str(v))
    return run_snakemake_target(target=target, snakefile="WES-Pipeline-Snakemake/workflow/Snakefile")
'''


def generate(registry_yaml: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    with open(registry_yaml, 'r') as f:
        reg = Registry.model_validate(yaml.safe_load(f))
    count = 0
    for tool in reg.tools.values():
        if not tool.outputs:
            continue
        tpl = tool.outputs[0].path_template or ""
        if not tpl:
            continue
        func_name = tool.rule.replace('-', '_')
        code = TEMPLATE.format(func_name=func_name, rule=tool.rule, target_template=tpl.replace('"','\"'))
        path = os.path.join(out_dir, f"{func_name}.py")
        with open(path, 'w') as f:
            f.write(code)
        count += 1
    # write __init__.py to export functions
    init_path = os.path.join(out_dir, "__init__.py")
    with open(init_path, 'w') as f:
        names = [t.rule.replace('-', '_') for t in reg.tools.values() if t.outputs]
        f.write("__all__ = [\n" + ",\n".join([f"    '{n}'" for n in names]) + "\n]\n")
    return count
