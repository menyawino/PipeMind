from __future__ import annotations
import os
from pipemind.registry.parser import parse_workflow_to_registry


MOCK_SNAKEFILE = """\nrule all:\n    input: expand("analysis/006_variant_filtering/{sample}.filtered.snp.vcf", sample=["S1","S2"])\n\nrule variant_filtering:\n    input:\n        vcf="analysis/005_calls/{sample}.vcf"\n    output:\n        filt="analysis/006_variant_filtering/{sample}.filtered.snp.vcf"\n    threads: 2\n    shell:\n        "echo FILTERED > {output.filt}"\n\nrule calls:\n    input:\n        raw="analysis/004_align/{sample}.bam"\n    output:\n        vcf="analysis/005_calls/{sample}.vcf"\n    shell:\n        "echo VCF > {output.vcf}"\n\n"""


def test_parse_rules(tmp_path):
    # Create mock workflow dir structure
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "Snakefile").write_text(MOCK_SNAKEFILE)
    out = tmp_path / "registry.yaml"
    reg = parse_workflow_to_registry(str(wf), str(out))
    # Expect two real rules (variant_filtering, calls)
    assert len(reg.tools) == 2, reg.tools.keys()
