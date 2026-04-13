from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from pipemind.mcp_server.server import create_app
from pipemind.registry.parser import parse_workflow_to_registry


DEFAULT_REPOS = [
    "snakemake-workflows/dna-seq-gatk-variant-calling",
    "snakemake-workflows/rna-seq-star-deseq2",
    "snakemake-workflows/rna-seq-kallisto-sleuth",
    "snakemake-workflows/chipseq",
    "vanheeringen-lab/seq2science",
]

DEFAULT_MIN_COMPILED_IMPORT_RATE = 0.80
DEFAULT_MIN_AGENT_READY_COVERAGE = 0.85


def clone_repo(repo: str, base_dir: Path) -> Path:
    repo_dir = base_dir / repo.split("/", 1)[1]
    if not (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(repo_dir)],
            check=True,
        )
    return repo_dir


def detect_workflow_dir(repo_dir: Path) -> Path:
    for candidate in (repo_dir / "workflow", repo_dir):
        if (candidate / "Snakefile").exists():
            return candidate
    for path in repo_dir.rglob("Snakefile"):
        return path.parent
    raise FileNotFoundError(f"No Snakefile found under {repo_dir}")


def evaluate_repo(repo: str, clone_root: Path) -> dict:
    repo_dir = clone_repo(repo, clone_root)
    workflow_dir = detect_workflow_dir(repo_dir)
    tempdir = Path(tempfile.mkdtemp(prefix=f"pipemind-eval-{repo_dir.name}-"))
    out_yaml = tempdir / "registry.yaml"

    summary = {
        "repo": repo,
        "repo_dir": str(repo_dir),
        "workflow_dir": str(workflow_dir),
    }
    try:
        reg = parse_workflow_to_registry(str(workflow_dir), str(out_yaml))
        parser_meta = reg.metadata.get("parser", {})
        issues = reg.metadata.get("issues", [])
        summary.update(
            {
                "parse_ok": True,
                "ingestion_mode": parser_meta.get("ingestion_mode"),
                "workdir": parser_meta.get("workdir"),
                "bootstrap_mode": parser_meta.get("bootstrap_mode"),
                "stubbed_imports": parser_meta.get("stubbed_imports", []),
                "shimmed_modules": parser_meta.get("shimmed_modules", []),
                "tool_count": parser_meta.get("tool_count", len(reg.tools)),
                "agent_ready_rule_count": parser_meta.get("agent_ready_rule_count"),
                "non_agent_ready_rule_count": parser_meta.get("non_agent_ready_rule_count"),
                "composition_ready_rule_count": parser_meta.get("composition_ready_rule_count"),
                "non_composition_ready_rule_count": parser_meta.get("non_composition_ready_rule_count"),
                "issue_count": len(issues),
                "issue_codes": sorted({issue.get("code", "unknown") for issue in issues}),
                "fallback_reason": parser_meta.get("fallback_reason"),
            }
        )
        try:
            parse_workflow_to_registry(str(workflow_dir), str(tempdir / "registry.strict.yaml"), strict=True)
            summary["strict_ok"] = True
        except Exception as exc:
            summary["strict_ok"] = False
            summary["strict_error"] = str(exc)
        try:
            api, mcp = create_app(str(out_yaml))
            summary["mcp_app_ok"] = True
            summary["mcp_tool_count"] = len(reg.tools) + 1
            summary["api_title"] = api.title
            summary["mcp_name"] = getattr(mcp, "name", None)
        except Exception as exc:
            summary["mcp_app_ok"] = False
            summary["mcp_error"] = str(exc)
    except Exception as exc:
        summary.update(
            {
                "parse_ok": False,
                "error": str(exc),
            }
        )
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
    return summary


def evaluate_many(repos: Iterable[str], clone_root: Path) -> list[dict]:
    return [evaluate_repo(repo, clone_root) for repo in repos]


def summarize_results(results: list[dict]) -> dict:
    total_repos = len(results)
    parsed_results = [result for result in results if result.get("parse_ok")]
    compiled_results = [result for result in parsed_results if result.get("ingestion_mode") == "compiled"]
    total_tools = sum(int(result.get("tool_count") or 0) for result in parsed_results)
    agent_ready_tools = sum(int(result.get("agent_ready_rule_count") or 0) for result in parsed_results)
    composition_ready_tools = sum(int(result.get("composition_ready_rule_count") or 0) for result in parsed_results)
    mcp_success_count = sum(1 for result in parsed_results if result.get("mcp_app_ok"))
    stubbed_compile_count = sum(1 for result in compiled_results if result.get("bootstrap_mode") == "import-stubbed")

    return {
        "repo_count": total_repos,
        "parse_success_count": len(parsed_results),
        "compiled_import_count": len(compiled_results),
        "compiled_import_rate": (len(compiled_results) / total_repos) if total_repos else 0.0,
        "stubbed_compile_count": stubbed_compile_count,
        "mcp_app_success_count": mcp_success_count,
        "mcp_app_success_rate": (mcp_success_count / len(parsed_results)) if parsed_results else 0.0,
        "tool_count": total_tools,
        "agent_ready_rule_count": agent_ready_tools,
        "agent_ready_coverage": (agent_ready_tools / total_tools) if total_tools else 0.0,
        "composition_ready_rule_count": composition_ready_tools,
        "composition_ready_coverage": (composition_ready_tools / total_tools) if total_tools else 0.0,
    }


def check_thresholds(summary: dict, min_compiled_import_rate: float, min_agent_ready_coverage: float) -> dict:
    checks = {
        "compiled_import_rate": summary.get("compiled_import_rate", 0.0) >= min_compiled_import_rate,
        "agent_ready_coverage": summary.get("agent_ready_coverage", 0.0) >= min_agent_ready_coverage,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PipeMind on public Snakemake workflows.")
    parser.add_argument("--clone-root", default="/tmp/pipemind-public-workflows", help="Directory for cloned workflow repositories")
    parser.add_argument("--min-compiled-import-rate", type=float, default=DEFAULT_MIN_COMPILED_IMPORT_RATE, help="Minimum fraction of evaluation repositories that must compile via Snakemake-native ingestion")
    parser.add_argument("--min-agent-ready-coverage", type=float, default=DEFAULT_MIN_AGENT_READY_COVERAGE, help="Minimum fraction of imported rules that must be agent-ready across the evaluation set")
    parser.add_argument("--no-enforce-thresholds", action="store_true", help="Report acceptance thresholds without failing the process when they are missed")
    parser.add_argument("repos", nargs="*", default=DEFAULT_REPOS, help="GitHub repositories in owner/name form")
    args = parser.parse_args()

    clone_root = Path(args.clone_root)
    clone_root.mkdir(parents=True, exist_ok=True)
    results = evaluate_many(args.repos, clone_root)
    summary = summarize_results(results)
    acceptance = check_thresholds(summary, args.min_compiled_import_rate, args.min_agent_ready_coverage)
    payload = {
        "evaluation_set": list(args.repos),
        "thresholds": {
            "min_compiled_import_rate": args.min_compiled_import_rate,
            "min_agent_ready_coverage": args.min_agent_ready_coverage,
        },
        "summary": summary,
        "acceptance": acceptance,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if not args.no_enforce_thresholds and not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()