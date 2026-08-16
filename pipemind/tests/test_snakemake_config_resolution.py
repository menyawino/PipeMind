import os, tempfile
from pipemind.snakemake.generator import generate_snakefile

REGISTRY = """
resources:
  cfg.outdir:
    id: cfg.outdir
    name: outdir
    resource_type: file
    uri: results_root
tools:
  snk.make_file:
    id: snk.make_file
    name: make_file
    rule: make_file
    description: test rule
    inputs: []
    outputs:
      - name: o
        io_type: txt
        path_template: '{config.outdir}/analysis/thing/{sample}.txt'
    params: []
    threads: 1
    command: 'echo HI > {config.outdir}/analysis/thing/{sample}.txt'
    log_paths: []
"""


def test_config_variable_resolution_in_goal_and_rule_all():
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td, "reg.yaml")
        with open(reg, "w", encoding="utf-8") as f:
            f.write(REGISTRY)
        snake = generate_snakefile(reg, ["{config.outdir}/analysis/thing/{sample}.txt"], {"sample": "A1"})
        assert "results_root/analysis/thing/A1.txt" in snake
        # Ensure no raw {config.outdir} placeholders linger
        assert "{config.outdir}" not in snake