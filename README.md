# PipeMind

PipeMind converts Snakemake workflows into an agent-usable bioinformatics toolset. It compiles a workflow into a typed registry, exposes compatible rules as MCP tools, preserves execution metadata needed for re-materialization, and builds minimal Snakemake plans for requested goal artifacts.

Highlights
- Snakemake-native ingestion uses the compiled workflow graph, not just local text scraping
- Parser converts rules -> strict typed tool registry (YAML) with agent-readiness diagnostics
- FastMCP server exposes tools/data with schemas for discovery and composition
- LLM intake app (CLI) asks for missing params and produces a fully bound config
- Dynamic DAG builder composes a minimal graph to achieve a goal artifact
- Audit log with inputs, params, hashes, and outputs for reproducibility
- Execution resolves Snakemake from the active interpreter, not only from an activated shell

Current conversion guarantees
- Imports rules from `include`, `module`, `use rule`, and `checkpoint` constructs when Snakemake can compile the workflow
- Preserves named inputs, outputs, params, logs, wrappers, scripts, containers, conda envs, resources, priorities, cache flags, and source locations
- Marks rules with dynamic callable IO or params as limited agent compatibility instead of silently treating them as static tools
- Falls back to recursive source scanning if compiled ingestion fails, and records that downgrade in registry metadata
- Supports `pipemind parse --strict` to fail fast when the workflow is not fully agent-ready

Quick start
- Provide the path to your Snakemake workflow directory when using the tool
- Install: pip install -e .
- Parse: pipemind parse <your-workflow-dir> -o pipemind/registry/registry.yaml
- Parse with full compatibility enforcement: pipemind parse <your-workflow-dir> -o pipemind/registry/registry.yaml --strict
- Serve schemas: pipemind serve --host 0.0.0.0 --port 8000
- Plan goal: pipemind goal --target-output "vcf:/analysis/006_variant_filtering/{sample}.filtered.snp.vcf" --fill '{"sample":"S1","lane":"L001","R":"R1"}'
- Run a target: pipemind run --target ../output_38/analysis/006_variant_filtering/S1.filtered.snp.vcf

Environment variables
- PIPEMIND_SNAKEFILE: Path to the Snakefile (e.g., <your-workflow-dir>/Snakefile)
- OPENAI_BASE_URL: Base URL for OpenAI-compatible servers (LM Studio, Ollama, etc.)
- PIPEMIND_LLM_MODEL: Model ID (e.g., llama-3.2-3b-instruct, qwen2.5:3b, gpt-4o-mini)
- OPENAI_API_KEY/OPENAI_API/OPENAI_KEY: API key for OpenAI-compatible servers
- PIPEMIND_OPENAI_KEY_FILE: Optional path to a file containing the API key; if unset, ./openai.api is used if present.

LLM client examples
- Local LM Studio (default port 1234): set OPENAI_BASE_URL=http://localhost:1234/v1 and choose a small CPU model; then run:
	pipemind llm "Summarize the pipeline steps."
- Ollama (default port 11434): set OPENAI_BASE_URL=http://localhost:11434/v1 and PIPEMIND_LLM_MODEL=llama3.2:3b-instruct

Packaging & distribution
- Package includes `pipemind/registry/*` by default to run the server out-of-the-box.
- Build wheel/sdist: python -m build
- Publish: twine upload dist/*

Design notes
- Registry schema uses Pydantic for strict typing and YAML as the exchange format.
- MCP server uses FastMCP to register each rule as a tool; schemas available at /schemas.
- DAG builder back-chains by matching IO types and explicit symbolic references; future work: richer semantic typing and constraints.
- CLI is built with Typer and prints JSON plans; integrate with any LLM agent for conversational UX.
- Registry metadata reports ingestion mode, source compatibility issues, and counts of fully agent-ready rules.

Landscape 2025-2026
- No 2025-2026 publication appears to do the same end-to-end job as PipeMind: compiling arbitrary Snakemake workflows into an MCP-addressable tool surface with re-composable execution plans.
- BioMaster (2025) is the closest bioinformatics system conceptually. It is a multi-agent workflow automation framework, but it orchestrates analyses at the agent layer rather than importing existing Snakemake ecosystems as reusable typed tools.
- MCPmed (2026) is the closest MCP-for-bioinformatics paper. It argues for MCP-enabled bioinformatics web services, but it is about machine-readable service backends, not workflow engines, rule provenance, or executable Snakemake composition.
- Many 2025-2026 Snakemake publications such as Colora, PopGLen, annoSnake, HaploCharmer, and MoGAAAP are pipeline papers, not workflow-to-agent translation layers. PipeMind complements them by turning such workflows into a uniform agent-facing tool abstraction instead of adding another fixed pipeline.
- MCP-AgentBench (2026) and 2025 orchestration reliability papers matter operationally: they show that tool-mediated agents fail on interface ambiguity and orchestration bugs. PipeMind should therefore emphasize strict parsing, explicit agent-readiness metadata, audit trails, and benchmarkable tool schemas.

Real-use gaps to keep closing
- Semantic typing is still mostly extension-based. Publication-grade deployment will benefit from ontology-backed artifact typing and richer constraints between tools.
- Dynamic Python callables in `input`, `params`, and `resources` are now surfaced explicitly, but they still require either user binding, source specialization, or sandboxed evaluation to become fully agent-callable.
- Evaluation should move beyond unit tests into benchmark tasks: import coverage on public Snakemake repositories, plan correctness, execution success, and agent success rates over MCP-mediated tasks.
- Interoperability should extend beyond Snakemake once the Snakemake path is solid: CWL, Nextflow, and WDL adapters are the natural expansion route, but they should not dilute the Snakemake-first correctness bar.

References/Citations
- Snakemake: Koster & Rahmann, Bioinformatics (2012)
- GATK Best Practices: Van der Auwera & O'Connor, O'Reilly (2020)
- FastAPI: https://fastapi.tiangolo.com/
- FastMCP (Model Context Protocol): https://github.com/AnswerDotAI/fastmcp
- Pydantic: https://docs.pydantic.dev/
- NetworkX: https://networkx.org/
 - Model Context Protocol Spec: https://spec.modelcontextprotocol.io/

How to add a new tool
- Add a new Snakemake rule or extend an existing .smk in your workflow's rules directory
- Re-run: pipemind parse <your-workflow-dir> -o pipemind/registry/registry.yaml
- The MCP server will automatically pick up the new tool upon restart

Security and reproducibility
- Access to private files/services is encoded in resources with access flags. Provide tokens via environment or secrets store.
- Every MCP tool call writes an audit record under `.pipemind/audit/` capturing parameters and outputs to enable re-runs.

Troubleshooting
- If the server fails at startup with a Pydantic error: ensure fastmcp>=0.4.1 is installed and re-install the project in your active virtualenv.
- If LLM calls fail with 429 or auth errors: verify OPENAI_BASE_URL and API key; for local, use LM Studio/Ollama and a small CPU-friendly model.
- If `pipemind parse --strict` fails: inspect `metadata.issues` in the generated registry to find dynamic rules or workflows that only parsed via source fallback.
