import os, tempfile
from pipemind.snakemake.generator import generate_snakefile

REGISTRY = """
tools:
  snk.split_vcfs:
    id: snk.split_vcfs
    name: split_vcfs
    rule: split_vcfs
    description: split
    inputs: []
    outputs:
      - name: snp_vcf
        io_type: vcf
        path_template: data/{sample}.snp.vcf
    params: []
    threads: 1
    command: 'echo SNP > data/{sample}.snp.vcf'
    log_paths: []
  snk.filter_snps:
    id: snk.filter_snps
    name: filter_snps
    rule: filter_snps
    description: filter
    inputs:
      - name: snp_vcf
        io_type: vcf
        path_template: rules.split_vcfs.output.snp_vcf
    outputs:
      - name: filtered_vcf
        io_type: vcf
        path_template: results/{sample}.filtered.snp.vcf
    params: []
    threads: 1
    command: 'cat data/{sample}.snp.vcf > results/{sample}.filtered.snp.vcf'
    log_paths: []
resources: {}
"""

def test_symbolic_reference_backchain():
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td,'reg.yaml')
        with open(reg,'w',encoding='utf-8') as f: f.write(REGISTRY)
        snake = generate_snakefile(reg, ['results/{sample}.filtered.snp.vcf'], {'sample':'AA'})
        # Both rules should appear in order (producer first or earlier in file)
        assert 'rule split_vcfs:' in snake
        assert 'rule filter_snps:' in snake
        assert "results/AA.filtered.snp.vcf" in snake