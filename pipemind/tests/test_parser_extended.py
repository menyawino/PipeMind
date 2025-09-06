from __future__ import annotations
from pipemind.registry.parser import _stringify, _split_sections


def test_stringify_unwraps_wrappers_and_expand():
    s1 = _stringify('directory("out/{sample}/file.txt")')
    assert s1 == "out/{sample}/file.txt"
    s2 = _stringify('expand("res/{x}.vcf", x=[1,2])')
    assert s2 == "res/{x}.vcf"
    s3 = _stringify('config["root"] + "/logs"')
    assert s3 == "{config.root}/logs"


def test_split_sections_handles_extended():
    block = '''
        input:
            a = "a.txt",
        resources:
            mem_mb = 4000,
        container: "docker://ubuntu:22.04"
        envmodules:
            "bwa",
            "samtools"
        run:
            """
            x = 1
            y = x + 2
            """
    '''
    secs = _split_sections(block)
    assert set(secs).issuperset({"input","resources","container","envmodules","run"})