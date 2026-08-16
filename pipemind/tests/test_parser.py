from __future__ import annotations
import os
import pytest
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
    assert reg.metadata["parser"]["ingestion_mode"] == "compiled"


def test_parse_workflow_handles_include(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "extra.smk").write_text(
        """
rule trim:
    input:
        'reads/{sample}.fq.gz'
    output:
        'trimmed/{sample}.fq.gz'
    shell:
        'cp {input} {output}'
""".strip()
    )
    (wf / "Snakefile").write_text("include: 'extra.smk'\n")

    reg = parse_workflow_to_registry(str(wf), str(tmp_path / "registry.yaml"))

    assert "snk.trim" in reg.tools
    assert reg.metadata["parser"]["ingestion_mode"] == "compiled"


def test_parse_workflow_handles_module_use_rule(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "module.smk").write_text(
        """
rule base:
    output:
        'module/{sample}.txt'
    shell:
        'echo base > {output}'
""".strip()
    )
    (wf / "Snakefile").write_text(
        """
module mod:
    snakefile: 'module.smk'

use rule * from mod as mod_*
""".strip()
    )

    reg = parse_workflow_to_registry(str(wf), str(tmp_path / "registry.yaml"))

    assert "snk.mod_base" in reg.tools
    assert reg.tools["snk.mod_base"].source_snakefile.endswith("module.smk")


def test_parse_workflow_flags_dynamic_inputs(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "Snakefile").write_text(
        """
def choose_reads(wc):
    return f'reads/{wc.sample}.fq.gz'

rule align:
    input:
        choose_reads
    output:
        'mapped/{sample}.bam'
    shell:
        'cp {input} {output}'
""".strip()
    )

    reg = parse_workflow_to_registry(str(wf), str(tmp_path / "registry.yaml"))
    tool = reg.tools["snk.align"]

    assert tool.agent_ready is True
    assert tool.composition_ready is False
    assert tool.inputs[0].path_template is None
    assert any(param.name == "bind_input_1" for param in tool.params)
    assert any(issue["code"] == "dynamic-input" for issue in reg.metadata["issues"])


def test_parse_workflow_strict_accepts_dynamic_rules_with_bindings(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "Snakefile").write_text(
        """
def choose_reads(wc):
    return f'reads/{wc.sample}.fq.gz'

rule align:
    input:
        choose_reads
    output:
        'mapped/{sample}.bam'
    shell:
        'cp {input} {output}'
""".strip()
    )

    reg = parse_workflow_to_registry(str(wf), str(tmp_path / "registry.yaml"), strict=True)

    assert reg.tools["snk.align"].agent_ready is True


def test_parse_workflow_bootstraps_missing_imports(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "Snakefile").write_text(
        """
import imaginary_frame_lib as ifl

SAMPLES = list(ifl.read_table('samples.tsv')['sample'])

rule imported:
    output:
        'results/{sample}.txt'
    shell:
        'echo ok > {output}'
""".strip()
    )

    reg = parse_workflow_to_registry(str(wf), str(tmp_path / "registry.yaml"))

    assert reg.metadata["parser"]["ingestion_mode"] == "compiled"
    assert reg.metadata["parser"]["bootstrap_mode"] == "import-stubbed"
    assert "imaginary_frame_lib" in reg.metadata["parser"]["stubbed_imports"]
    assert any(issue["code"] == "workflow-import-stubbed" for issue in reg.metadata["issues"])


def test_parse_workflow_shims_missing_module_attributes(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "xdg.py").write_text("CACHE = '/tmp'\n")
    (wf / "Snakefile").write_text(
        """
import xdg

XDG_CACHE_HOME = xdg.XDG_CACHE_HOME

rule shimmed:
    output:
        'results/{sample}.txt'
    shell:
        'echo ok > {output}'
""".strip()
    )

    reg = parse_workflow_to_registry(str(wf), str(tmp_path / "registry.yaml"))

    assert reg.metadata["parser"]["ingestion_mode"] == "compiled"
    assert reg.metadata["parser"]["bootstrap_mode"] == "import-stubbed"
    assert "xdg" in reg.metadata["parser"]["shimmed_modules"]
    assert any(issue["code"] == "workflow-module-shimmed" for issue in reg.metadata["issues"])


def test_parse_workflow_dir_uses_repo_root_for_standard_layout(tmp_path):
    repo = tmp_path / "repo"
    workflow = repo / "workflow"
    config_dir = repo / "config"
    workflow.mkdir(parents=True)
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("answer: 42\n")
    (workflow / "Snakefile").write_text(
        """
configfile: "config/config.yaml"

rule answer:
    output:
        "results/answer.txt"
    shell:
        "echo {config[answer]} > {output}"
""".strip()
    )

    reg = parse_workflow_to_registry(str(workflow), str(tmp_path / "registry.yaml"))

    assert reg.metadata["parser"]["ingestion_mode"] == "compiled"
    assert reg.metadata["parser"]["workdir"] == str(repo.resolve())
    assert any(path.endswith("config/config.yaml") for path in reg.metadata["parser"]["configfiles"])
    assert "snk.answer" in reg.tools


def test_parse_workflow_flags_dynamic_resources_without_fallback(tmp_path):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "Snakefile").write_text(
        """
rule dynamic_mem:
    output:
        'results/{sample}.txt'
    resources:
        mem_mb=lambda wildcards: 2000
    shell:
        'echo ok > {output}'
""".strip()
    )

    reg = parse_workflow_to_registry(str(wf), str(tmp_path / "registry.yaml"))

    assert reg.metadata["parser"]["ingestion_mode"] == "compiled"
    assert any(issue["code"] == "dynamic-resource" for issue in reg.metadata["issues"])
    assert "snk.dynamic_mem" in reg.tools
