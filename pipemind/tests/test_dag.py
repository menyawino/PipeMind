from __future__ import annotations
import os
from pipemind.registry.parser import parse_workflow_to_registry
from pipemind.dag.builder import build_dag_for_goal


def test_build_dag(tmp_path):
    wf = "/path/to/your/workflow"
    reg_path = tmp_path / "registry.yaml"
    parse_workflow_to_registry(wf, str(reg_path))
    plan = build_dag_for_goal(str(reg_path), "vcf:/analysis/006_variant_filtering/{sample}.filtered.snp.vcf", {"sample":"S1","lane":"L001","R":"R1"})
    assert plan["steps"], plan
