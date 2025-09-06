from __future__ import annotations
from typing import Optional
import os


def guess_io_type(path: str) -> str:
    """Best-effort semantic IO type from file extension.

    Examples:
    - .fastq -> fastq
    - .fastq.gz -> fastq_gz
    - .g.vcf or .g.vcf.gz -> gvcf or gvcf_gz
    """
    p = path.lower()
    if p.endswith(".fastq.gz"):
        return "fastq_gz"
    if p.endswith(".fastq"):
        return "fastq"
    if p.endswith(".bam.bai"):
        return "bam_bai"
    if p.endswith(".bam"):
        return "bam"
    if p.endswith(".cram.crai"):
        return "crai"
    if p.endswith(".cram"):
        return "cram"
    if p.endswith(".bed.gz"):
        return "bed_gz"
    if p.endswith(".g.vcf.gz"):
        return "gvcf_gz"
    if p.endswith(".g.vcf"):
        return "gvcf"
    if p.endswith(".vcf.gz"):
        return "vcf_gz"
    if p.endswith(".vcf"):
        return "vcf"
    if p.endswith(".tbi"):
        return "tbi"
    if p.endswith(".bed"):
        return "bed"
    if p.endswith(".idx"):
        return "idx"
    if p.endswith(".gzi"):
        return "gzi"
    if p.endswith(".fasta"):
        return "fasta"
    if p.endswith(".fa"):
        return "fa"
    if p.endswith(".fai"):
        return "fai"
    if p.endswith(".dict"):
        return "dict"
    if p.endswith(".html"):
        return "html"
    if p.endswith(".png"):
        return "png"
    if p.endswith(".zip"):
        return "zip"
    if p.endswith(".json"):
        return "json"
    if p.endswith(".csv"):
        return "csv"
    if p.endswith(".tsv"):
        return "tsv"
    if p.endswith(".txt"):
        return "txt"
    if p.endswith("/") or os.path.isdir(path):
        return "directory"
    return "unknown"


def snakemake_wildcards(path: str) -> set[str]:
    """Return set of Snakemake wildcards like {sample} present in the path template."""
    import re

    return set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", path))
