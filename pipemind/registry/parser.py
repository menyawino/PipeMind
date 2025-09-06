from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import os
import re
import yaml

from .schema import IODecl, ParamDecl, ToolSpec, Registry, ResourceSpec
from pipemind.utils.io_types import guess_io_type


# Match both rule and checkpoint blocks
RULE_RE = re.compile(r"^(rule|checkpoint)\s+([a-zA-Z0-9_]+):", re.MULTILINE)


def _stringify(expr: str) -> str:
    """Normalize a Snakemake Python-like expression into a compact path template.

    - Convert config["key"] -> {config.key}
    - Remove string quotes and '+' concatenations without leaving spaces
    - Collapse whitespace and remove spaces around slashes
    - Extract the first template from expand("...", ...) if present
    """
    s = expr.strip()

    # Pull out expand("...", ...) or expand(["...", ...], ...) template if present (keep inside string only)
    # Replace iteratively in case of nesting
    expand_re = re.compile(r"expand\s*\(\s*(['\"])\s*(.*?)\s*\1(?:\s*,[^)]*)?\)", re.DOTALL)
    expand_list_re = re.compile(r"expand\s*\(\s*\[(.*?)\](?:\s*,[^)]*)?\)", re.DOTALL)
    while True:
        m = expand_re.search(s)
        ml = expand_list_re.search(s) if not m else None
        if not m and not ml:
            break
        if m:
            tpl = m.group(2)
            s = s[:m.start()] + tpl + s[m.end():]
        elif ml:
            # pick the first quoted string inside list
            inner = ml.group(1)
            q = re.search(r"(['\"])\s*(.*?)\s*\1", inner)
            tpl = q.group(2) if q else inner
            s = s[:ml.start()] + tpl + s[ml.end():]

    # Replace config["key"] and config.get('key', ...) with {config.key} using named groups
    s = re.sub(r'config\["(?P<key>[^\"]+)"\]', lambda m: f"{{config.{m.group('key')}}}", s)
    s = re.sub(r"config\['(?P<key>[^']+)'\]", lambda m: f"{{config.{m.group('key')}}}", s)
    s = re.sub(r"config\.get\(\s*['\"](?P<key>[^'\"]+)['\"](?:\s*,[^)]*)?\)", lambda m: f"{{config.{m.group('key')}}}", s)
    # Collapse chained indices like {config.a}["b"]["c"] -> {config.a.b.c}
    chain_re = re.compile(r"(\{config\.[^}]*?)\}\s*\[\s*['\"]([^'\"]+)['\"]\s*\]")
    while True:
        m = chain_re.search(s)
        if not m:
            break
        start = m.start(); end = m.end(); g1 = m.group(1); g2 = m.group(2)
        s = s[:start] + g1 + "." + g2 + "}" + s[end:]

    # os.path.join("a","b", x) -> a/b/x (naive join)
    join_re = re.compile(r"os\.path\.join\(\s*(.*?)\s*\)", re.DOTALL)
    def _join_repl(m):
        args = m.group(1)
        parts = [p.strip() for p in args.split(',') if p.strip()]
        # Remove surrounding quotes from each part (quotes also removed globally later)
        parts = [re.sub(r'^[\'\"]|[\'\"]$', '', p) for p in parts]
        return "/".join(parts)
    s = join_re.sub(_join_repl, s)

    # Unwrap common Snakemake wrappers that wrap a single path
    # e.g., directory("a/{x}"), temp("..."), protected("..."), touch("...")
    wrap_names = [
        "directory", "temp", "protected", "touch", "ancient", "dynamic",
    ]
    for wn in wrap_names:
        pattern = re.compile(rf"{re.escape(wn)}\(\s*(?P<quote>['\"])\s*(?P<content>.*?)\s*(?P=quote)\s*\)")
        s = pattern.sub(lambda m: m.group('content'), s)

    # Remove quotes
    s = s.replace("'", "").replace('"', '')

    # Remove stray f-prefix before templates (from f"..." strings)
    s = re.sub(r"\bf(?=\{)", "", s)

    # Remove concatenation operators and surrounding whitespace
    s = re.sub(r"\s*\+\s*", "", s)

    # Remove trailing commas
    s = s.rstrip(',')

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    s = s.strip()

    # Remove spaces around slashes (e.g., '{config.outdir} /logs' -> '{config.outdir}/logs')
    s = re.sub(r"\s*/\s*", "/", s)

    return s


Section = Tuple[str, List[str]]  # (inline_value, block_lines)


def _split_sections(rule_block: str) -> Dict[str, Section]:
    """Split a rule body into top-level sections keyed by name (input/output/etc).

    We detect headers at the first indentation level within the rule body and
    then capture all subsequent lines verbatim until the next header at that
    same level. This preserves blank lines and triple-quoted blocks fully.
    """
    known = {
        "input",
        "output",
        "params",
        "log",
        "threads",
        "conda",
        "resources",
        "shell",
        "script",
        "message",
        "benchmark",
        "container",
        "envmodules",
        "wrapper",
        "run",
        "group",
        "priority",
        "cache",
    }
    lines = rule_block.splitlines()
    sections: Dict[str, Section] = {}
    header_indent: Optional[str] = None
    current_key: Optional[str] = None
    current_inline: str = ""
    current_lines: List[str] = []

    header_re = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?P<inline>.*)$")

    def flush():
        nonlocal current_key, current_inline, current_lines
        if current_key is not None:
            sections[current_key] = (current_inline, current_lines[:])
        current_key = None
        current_inline = ""
        current_lines = []

    for line in lines:
        m = header_re.match(line)
        if m and m.group("key") in known:
            indent = m.group("indent")
            key = m.group("key")
            inline = m.group("inline").strip()
            # Record the first header indent as the canonical header level
            if header_indent is None:
                header_indent = indent
            # If this is a new header at the header level, start a new section
            if indent == header_indent:
                flush()
                current_key = key
                current_inline = inline
                continue
        # If not a new header at header level and we're inside a section, capture verbatim
        if current_key is not None:
            current_lines.append(line)
            continue
        # Outside any section, ignore stray lines (comments/whitespace)
        continue
    # Flush last
    flush()
    return sections


def _kv_lines(inline: str, block_lines: List[str]) -> Dict[str, str]:
    """Parse key=value lines from a section. If the section uses a single inline
    scalar (e.g., input: file), capture that as an anonymous entry.
    """
    out: Dict[str, str] = {}
    # Handle inline singletons
    if inline:
        s = inline.strip().rstrip(',')
        if s:
            out["_1"] = s
    for raw in block_lines:
        line = raw.strip().rstrip(',')
        if not line or line.startswith('#'):
            continue
        if '=' in line and not line.startswith('|'):
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
        else:
            out[f"_{len(out)+1}"] = line
    return out


def parse_rules_file(path: str, cfg: Optional[Dict[str, Any]] = None, keep_shell_comments: bool = True) -> Dict[str, ToolSpec]:
    with open(path, 'r') as f:
        text = f.read()

    tools: Dict[str, ToolSpec] = {}
    for match in RULE_RE.finditer(text):
        kind = match.group(1)
        name = match.group(2)
        # Skip the aggregator rule
        if name == "all":
            continue
        # Get rule block by slicing until next rule or end
        start = match.end()
        next_m = RULE_RE.search(text, start)
        rule_block = text[start: (next_m.start() if next_m else len(text))]
        sections = _split_sections(rule_block)

        in_inline, in_lines = sections.get('input', ("", []))
        out_inline, out_lines = sections.get('output', ("", []))
        prm_inline, prm_lines = sections.get('params', ("", []))
        log_inline, log_lines = sections.get('log', ("", []))
        thr_inline, thr_lines = sections.get('threads', ("", []))
        res_inline, res_lines = sections.get('resources', ("", []))
        con_inline, con_lines = sections.get('conda', ("", []))
        ctr_inline, ctr_lines = sections.get('container', ("", []))
        mod_inline, mod_lines = sections.get('envmodules', ("", []))
        sh_inline, sh_lines = sections.get('shell', ("", []))
        sc_inline, sc_lines = sections.get('script', ("", []))
        wrp_inline, wrp_lines = sections.get('wrapper', ("", []))
        run_inline, run_lines = sections.get('run', ("", []))
        msg_inline, msg_lines = sections.get('message', ("", []))
        bmk_inline, bmk_lines = sections.get('benchmark', ("", []))
        grp_inline, grp_lines = sections.get('group', ("", []))
        pri_inline, pri_lines = sections.get('priority', ("", []))
        cac_inline, cac_lines = sections.get('cache', ("", []))

        inputs: List[IODecl] = []
        for k, v in _kv_lines(in_inline, in_lines).items():
            s = _stringify(v)
            inputs.append(IODecl(name=k, io_type=guess_io_type(s), path_template=s))

        outputs: List[IODecl] = []
        for k, v in _kv_lines(out_inline, out_lines).items():
            s = _stringify(v)
            outputs.append(IODecl(name=k, io_type=guess_io_type(s), path_template=s))

        params: List[ParamDecl] = []
        for k, v in _kv_lines(prm_inline, prm_lines).items():
            s = _stringify(v)
            params.append(ParamDecl(name=k, description=s))

        logs: List[str] = []
        for _, v in _kv_lines(log_inline, log_lines).items():
            logs.append(_stringify(v))

        def _parse_threads(inline: str, lines: List[str]) -> Optional[int]:
            raw = (inline or "").strip()
            if not raw and lines:
                raw = lines[0].strip()
            if not raw:
                return None
            # Try direct integer
            m = re.search(r"(\d+)", raw)
            if m and m.group(1):
                try:
                    return int(m.group(1))
                except Exception:
                    pass
            # Try to resolve config["threads_*"] from cfg
            if cfg is not None:
                m2 = re.search(r'config\["([^"]+)"\]', raw)
                if m2:
                    key = m2.group(1)
                    try:
                        val = cfg.get(key)
                        if isinstance(val, int):
                            return val
                        if isinstance(val, str) and val.isdigit():
                            return int(val)
                    except Exception:
                        pass
            return None

        threads = _parse_threads(thr_inline, thr_lines)

        def _parse_scalar(inline: str, lines: List[str]) -> Optional[str]:
            raw = (inline or "").strip()
            if not raw and lines:
                # take the first non-empty line
                for ln in lines:
                    s = ln.strip()
                    if s:
                        raw = s
                        break
            if not raw:
                return None
            # Strip quotes and trailing commas
            raw = raw.strip().strip(",").strip('"').strip("'")
            return raw or None

        conda_env = _parse_scalar(con_inline, con_lines)

        # Resources: key=value pairs, attempt to cast to int/float/bool, resolve config keys
        def _cast_value(val: str) -> Any:
            s = val.strip()
            # Handle config["key"]
            m = re.search(r'config\[[\'\"]([^\'\"]+)[\'\"]\]', s)
            if m and cfg is not None:
                key = m.group(1)
                return cfg.get(key, s)
            # Try bool
            if s.lower() in {"true", "false"}:
                return s.lower() == "true"
            # Try int
            if re.fullmatch(r"-?\d+", s):
                try:
                    return int(s)
                except Exception:
                    return s
            # Try float
            if re.fullmatch(r"-?\d+\.\d+", s):
                try:
                    return float(s)
                except Exception:
                    return s
            # Strip quotes and trailing commas
            s2 = s.strip().strip(',').strip('"').strip("'")
            return s2

        resources: Dict[str, Any] = {}
        for k, v in _kv_lines(res_inline, res_lines).items():
            resources[k] = _cast_value(v)

        # Shell command: flatten multi-line text into a single line (remove \ line continuations and newlines)
        command = None
        if sh_inline or sh_lines:
            raw_lines: List[str] = []
            if sh_inline and sh_inline not in {'"""', "'''"}:
                raw_lines.append(sh_inline)
            raw_lines.extend([ln.rstrip('\n') for ln in sh_lines])
            # Trim leading/trailing blank lines only
            while raw_lines and raw_lines[0].strip() == "":
                raw_lines.pop(0)
            while raw_lines and raw_lines[-1].strip() == "":
                raw_lines.pop()
            # Drop leading/trailing triple-quote delimiter lines
            if raw_lines and raw_lines[0].strip() in {'"""', "'''"}:
                raw_lines = raw_lines[1:]
            if raw_lines and raw_lines[-1].strip() in {'"""', "'''"}:
                raw_lines = raw_lines[:-1]
            # Optionally remove comment-only lines
            if not keep_shell_comments:
                raw_lines = [rl for rl in raw_lines if not re.match(r"^\s*#", rl)]
            # Remove line-continuation backslashes at EOL and flatten to a single line
            processed = []
            for rl in raw_lines:
                # Strip trailing backslash used for line continuation
                rl = re.sub(r"\\\s*$", "", rl)
                processed.append(rl.strip())
            # Join with single spaces and collapse redundant spaces
            cmd = " ".join([p for p in processed if p != ""]).strip()
            cmd = re.sub(r"\s+", " ", cmd)
            command = cmd if cmd else None

        script = None
        sc = _parse_scalar(sc_inline, sc_lines)
        if sc:
            m = re.search(r"([\w\-/\.]+\.(py|sh|R))", sc)
            script = m.group(1) if m else sc

        wrapper = None
        wr = _parse_scalar(wrp_inline, wrp_lines)
        if wr:
            wrapper = wr

        container = None
        container_engine = None
        ct = _parse_scalar(ctr_inline, ctr_lines)
        if ct:
            container = ct
            if ct.startswith("docker://"):
                container_engine = "docker"
            elif ct.startswith("shub://") or ct.startswith("library://") or ct.endswith(".sif"):
                container_engine = "singularity"

        # envmodules: collect a list of module names from inline or block
        envmodules: List[str] = []
        if mod_inline or mod_lines:
            raw = (mod_inline or "").strip()
            lines = [ln.strip() for ln in mod_lines if ln.strip()]
            joined = raw
            if lines:
                joined = (joined + "," if joined else "") + ",".join(lines)
            # Split by comma or whitespace and strip quotes
            toks = re.split(r"[\s,]+", joined)
            envmodules = [re.sub(r'^[\"\']|[\"\']$', '', t) for t in toks if t]

        # Optional: message and benchmark
        message = None
        if msg_inline or msg_lines:
            # take inline first, else join block
            txt = msg_inline if msg_inline else "\n".join([l.strip() for l in msg_lines if l.strip()])
            message = txt or None
        benchmark = _parse_scalar(bmk_inline, bmk_lines)

        # run: capture raw python block as text (flatten similar to shell but keep newlines)
        run_code = None
        if run_inline or run_lines:
            raw_lines: List[str] = []
            if run_inline and run_inline not in {'"""', "'''"}:
                raw_lines.append(run_inline)
            raw_lines.extend([ln.rstrip('\n') for ln in run_lines])
            # Trim surrounding triple-quote delimiters
            if raw_lines and raw_lines[0].strip() in {'"""', "'''"}:
                raw_lines = raw_lines[1:]
            if raw_lines and raw_lines[-1].strip() in {'"""', "'''"}:
                raw_lines = raw_lines[:-1]
            run_code = "\n".join(raw_lines).strip() or None

        # group, priority, cache
        group = _parse_scalar(grp_inline, grp_lines)
        priority = None
        pr = _parse_scalar(pri_inline, pri_lines)
        if pr and pr.isdigit():
            priority = int(pr)
        cache = None
        cc = _parse_scalar(cac_inline, cac_lines)
        if cc is not None:
            if cc.lower() in {"true", "false"}:
                cache = cc.lower() == "true"

        tool = ToolSpec(
            id=f"snk.{name}",
            name=name,
            rule=name,
            description=f"Snakemake rule {name}",
            message=message,
            inputs=inputs,
            outputs=outputs,
            params=params,
            threads=threads,
            conda_env=conda_env,
            command=command,
            script=script,
            benchmark=benchmark,
            log_paths=logs,
            resources=resources,
            container=container,
            container_engine=container_engine,
            envmodules=envmodules,
            wrapper=wrapper,
            run_code=run_code,
            group=group,
            priority=priority,
            cache=cache,
        )
        tools[tool.id] = tool

    return tools


def parse_workflow_to_registry(workflow_dir: str, out_yaml: str, keep_shell_comments: bool = True) -> Registry:
    # gather all .smk files
    rules_dir = os.path.join(workflow_dir, "rules")
    snakefile = os.path.join(workflow_dir, "Snakefile")
    tools: Dict[str, ToolSpec] = {}
    resources: Dict[str, ResourceSpec] = {}

    # Load config first (used for resolving threads and resources)
    cfg: Dict[str, Any] = {}
    cfg_path = os.path.join(workflow_dir, "config.yml")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as cf:
            cfg = yaml.safe_load(cf) or {}

    # Parse included rule files
    if os.path.isdir(rules_dir):
        for fn in sorted(os.listdir(rules_dir)):
            if fn.endswith('.smk'):
                tools.update(parse_rules_file(os.path.join(rules_dir, fn), cfg, keep_shell_comments))
    # Also parse top-level Snakefile in case it defines rules
    if os.path.exists(snakefile):
        tools.update(parse_rules_file(snakefile, cfg, keep_shell_comments))

    # Ingest config.yml as resources if present
    if cfg:
        for k, v in cfg.items():
            if isinstance(v, str) and ('/' in v or v.startswith('http')):
                rid = f"cfg.{k}"
                resources[rid] = ResourceSpec(
                    id=rid,
                    name=k,
                    resource_type="file" if not str(v).startswith("http") else "service",
                    uri=str(v),
                    access="private" if str(v).startswith("/mnt/") or str(v).startswith("/data/") else "public",
                    description=f"Config resource {k}",
                )

    reg = Registry(tools=tools, resources=resources)
    with open(out_yaml, 'w') as f:
        yaml.safe_dump(reg.model_dump(), f, sort_keys=True)
    return reg
