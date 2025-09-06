from __future__ import annotations
from typing import Dict, Any
import subprocess


def run_snakemake_target(target: str, snakefile: str, cores: int = 4) -> Dict[str, Any]:
    cmd = ["snakemake", "-s", snakefile, target, "-c", str(cores), "--rerun-incomplete", "--printshellcmds"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": res.returncode,
        "stdout": res.stdout[-4000:],
        "stderr": res.stderr[-4000:],
    }
