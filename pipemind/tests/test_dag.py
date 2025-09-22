from __future__ import annotations
import os
from pipemind.registry.parser import parse_workflow_to_registry
from pipemind.dag.builder import build_dag_for_goal

MOCK_SNAKEFILE = """\nrule all:\n    input: expand("analysis/006_variant_filtering/{sample}.filtered.snp.vcf", sample=["S1"])\n\nrule variant_filtering:\n    input:\n        vcf="analysis/005_calls/{sample}.vcf"\n    output:\n        filt="analysis/006_variant_filtering/{sample}.filtered.snp.vcf"\n    threads: 2\n    shell:\n        "echo FILTERED > {output.filt}"\n\nrule calls:\n    input:\n        raw="analysis/004_align/{sample}.bam"\n    output:\n        vcf="analysis/005_calls/{sample}.vcf"\n    shell:\n        "echo VCF > {output.vcf}"\n\n"""


def test_build_dag(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "Snakefile").write_text(MOCK_SNAKEFILE)
    reg_path = tmp_path / "registry.yaml"
    parse_workflow_to_registry(str(wf), str(reg_path))
    # Goal by io type prefix (vcf:) should back-chain to variant_filtering rule
    plan = build_dag_for_goal(str(reg_path), "vcf:analysis/006_variant_filtering/{sample}.filtered.snp.vcf", {"sample":"S1"})
    assert plan["steps"], plan
