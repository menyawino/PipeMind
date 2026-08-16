from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import os
import shutil
import subprocess
import sys


def resolve_snakemake_command() -> list[str]:
    explicit = os.getenv("PIPEMIND_SNAKEMAKE")
    if explicit:
        return [explicit]

    snakemake_bin = shutil.which("snakemake")
    if snakemake_bin:
        return [snakemake_bin]

    sibling = Path(sys.executable).with_name("snakemake")
    if sibling.exists():
        return [str(sibling)]

    return [sys.executable, "-m", "snakemake"]


def run_snakemake_target(target: str, snakefile: str, cores: int = 4, cwd: str | None = None) -> Dict[str, Any]:
    cmd = [*resolve_snakemake_command(), "-s", snakefile, target, "-c", str(cores), "--rerun-incomplete", "--printshellcmds"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return {
        "returncode": res.returncode,
        "stdout": res.stdout[-4000:],
        "stderr": res.stderr[-4000:],
        "cmd": cmd,
    }
