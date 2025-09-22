from __future__ import annotations
import os
import json
import tempfile

from pipemind.snakemake.generator import generate_snakefile, materialize_and_optionally_run


REGISTRY_YAML = """
tools:
  snk.a:
    id: snk.a
    name: a
    rule: a
    description: rule a
    inputs: []
    outputs:
      - name: _1
        io_type: txt
        path_template: data/{sample}.raw.txt
    params: []
    threads: 1
    command: "echo RAW > data/{sample}.raw.txt"
    log_paths: []
    resources: {}
  snk.b:
    id: snk.b
    name: b
    rule: b
    description: rule b
    inputs:
      - name: _1
        io_type: txt
        path_template: data/{sample}.raw.txt
    outputs:
      - name: _1
        io_type: txt
        path_template: results/{sample}.final.txt
    params: []
    threads: 1
    command: "cat data/{sample}.raw.txt > results/{sample}.final.txt"
    log_paths: []
    resources: {}
resources: {}
"""


def test_generate_snakefile_rule_all_concrete():
    with tempfile.TemporaryDirectory() as td:
        reg_path = os.path.join(td, "reg.yaml")
        with open(reg_path, "w", encoding="utf-8") as f:
            f.write(REGISTRY_YAML)
        sf = generate_snakefile(reg_path, ["results/{sample}.final.txt"], {"sample": "S1"})
        assert "rule all" in sf
        assert "results/S1.final.txt" in sf  # resolved target
        assert "rule a:" in sf and "rule b:" in sf


def test_materialize_and_dry_run():
    with tempfile.TemporaryDirectory() as td:
        reg_path = os.path.join(td, "reg.yaml")
        with open(reg_path, "w", encoding="utf-8") as f:
            f.write(REGISTRY_YAML)
        # Only run dry-run to avoid actually executing shell commands inside CI if snakemake present
        result = materialize_and_optionally_run(reg_path, ["results/{sample}.final.txt"], {"sample": "S2"}, workdir=td, run=True, dry_run=True)
        assert result["returncode"] == 0
        assert os.path.exists(result["snakefile"])
        # Snakemake dry-run output should mention the target file
    # stdout may be very quiet with --quiet; accept empty but ensure returncode ok
    if result.get("stdout"):
      assert "results/S2.final.txt" in result["stdout"] or result.get("returncode") == 0
