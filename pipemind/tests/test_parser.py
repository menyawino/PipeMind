from __future__ import annotations
import os
from pipemind.registry.parser import parse_workflow_to_registry


def test_parse_rules(tmp_path):
    wf = "/home/omar/bucket/WES-Pipeline-Snakemake/workflow"
    out = tmp_path / "registry.yaml"
    reg = parse_workflow_to_registry(wf, str(out))
    assert len(reg.tools) > 5
