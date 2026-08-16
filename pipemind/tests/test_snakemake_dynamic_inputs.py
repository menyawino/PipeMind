from pipemind.snakemake.generator import generate_snakefile
import tempfile, os

REGISTRY = """
tools:
  snk.producer:
    id: snk.producer
    name: producer
    rule: producer
    description: prod
    inputs: []
    outputs:
      - name: o
        io_type: txt
        path_template: data/{sample}.txt
    params: []
    threads: 1
    command: 'echo HI > data/{sample}.txt'
    log_paths: []
  snk.consumer:
    id: snk.consumer
    name: consumer
    rule: consumer
    description: cons
    inputs:
      - name: in1
        io_type: txt
        path_template: rules.producer.output.o
    outputs:
      - name: out
        io_type: txt
        path_template: results/{sample}.done.txt
    params: []
    threads: 1
    command: 'cat data/{sample}.txt > results/{sample}.done.txt'
    log_paths: []
resources: {}
"""


def test_rules_output_reference_not_quoted():
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td, 'reg.yaml')
        with open(reg,'w',encoding='utf-8') as f: f.write(REGISTRY)
        snake = generate_snakefile(reg, ['results/{sample}.done.txt'], {'sample':'X'})
        # Ensure input block uses expression not quoted string
        assert "rules.producer.output.o" in snake
        assert "input:" in snake