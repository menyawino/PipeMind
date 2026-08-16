from __future__ import annotations
from contextlib import contextmanager
import importlib.abc
import importlib
import importlib.machinery
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Iterable
import os
import re
import sys
import types
import yaml

from .schema import IODecl, ParamDecl, ToolSpec, Registry, ResourceSpec
from pipemind.utils.io_types import guess_io_type


# Match both rule and checkpoint blocks
RULE_RE = re.compile(r"^(rule|checkpoint)\s+([a-zA-Z0-9_]+):", re.MULTILINE)
_STUBBABLE_IMPORT_LIMIT = 24


class _StubValue:
    def __init__(self, name: str = "stub"):
        self._name = name

    def __getattr__(self, attr: str) -> "_StubValue":
        return _StubValue(f"{self._name}.{attr}")

    def __call__(self, *args: Any, **kwargs: Any) -> "_StubValue":
        return _StubValue(f"{self._name}()")

    def __getitem__(self, key: Any) -> "_StubValue":
        return _StubValue(f"{self._name}[{key!r}]")

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        return False

    def __contains__(self, item: Any) -> bool:
        return False

    def __str__(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"<StubValue {self._name}>"

    def __fspath__(self) -> str:
        return "/dev/null"

    def __mro_entries__(self, bases: tuple[type, ...]) -> tuple[type, ...]:
        return ()

    def __int__(self) -> int:
        return 0

    def __float__(self) -> float:
        return 0.0

    def __eq__(self, other: Any) -> bool:
        return False

    def __lt__(self, other: Any) -> bool:
        return False

    def __le__(self, other: Any) -> bool:
        return False

    def __gt__(self, other: Any) -> bool:
        return False

    def __ge__(self, other: Any) -> bool:
        return False

    def __add__(self, other: Any) -> Any:
        return other if isinstance(other, str) else self

    def __radd__(self, other: Any) -> Any:
        return other if isinstance(other, str) else self

    def __or__(self, other: Any) -> "_StubValue":
        return self

    def __ror__(self, other: Any) -> Any:
        return other if isinstance(other, dict) else self

    def keys(self) -> list[Any]:
        return []

    def items(self) -> list[tuple[Any, Any]]:
        return []

    def values(self) -> list[Any]:
        return []

    def get(self, key: Any, default: Any = None) -> Any:
        return default

    def copy(self) -> "_StubValue":
        return self

    def unique(self) -> "_StubValue":
        return self

    def tolist(self) -> list[Any]:
        return []

    def to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def itertuples(self, *args: Any, **kwargs: Any):
        return iter(())

    def iterrows(self):
        return iter(())


class _StubModule(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__all__ = []
        self.__path__ = []

    def __getattr__(self, attr: str) -> _StubValue:
        return _StubValue(f"{self.__name__}.{attr}")


class _WorkflowImportStubber(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, modules: Iterable[str]):
        self.modules = set(modules)
        self.loaded_modules: set[str] = set()

    def _matches(self, fullname: str) -> bool:
        return any(fullname == mod or fullname.startswith(mod + ".") for mod in self.modules)

    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if not self._matches(fullname):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        module = _StubModule(spec.name)
        self.loaded_modules.add(spec.name)
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        return None


@contextmanager
def _stub_workflow_imports(modules: Iterable[str]):
    stubber = _WorkflowImportStubber(modules)
    original_meta_path = list(sys.meta_path)
    original_modules = set(sys.modules)
    sys.meta_path.insert(0, stubber)
    try:
        yield stubber
    finally:
        sys.meta_path[:] = original_meta_path
        for name in sorted(stubber.loaded_modules, key=len, reverse=True):
            if name not in original_modules:
                sys.modules.pop(name, None)


def _extract_missing_module(exc: ModuleNotFoundError) -> Optional[str]:
    missing = getattr(exc, "name", None)
    if missing:
        return str(missing)
    match = re.search(r"No module named ['\"]([^'\"]+)['\"]", str(exc))
    if match:
        return match.group(1)
    return None


def _extract_missing_module_attribute(exc: Exception) -> Optional[str]:
    match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute ['\"]([^'\"]+)['\"]", str(exc))
    if match:
        return match.group(1)
    return None


@contextmanager
def _shim_workflow_modules(modules: Iterable[str]):
    patched: List[tuple[types.ModuleType, bool, Any]] = []
    try:
        for name in modules:
            module = importlib.import_module(name)
            had_getattr = hasattr(module, "__getattr__")
            original_getattr = getattr(module, "__getattr__", None)

            def _shim(attr: str, _module_name: str = name, _original: Any = original_getattr):
                if _original is not None:
                    try:
                        return _original(attr)
                    except AttributeError:
                        pass
                return _StubValue(f"{_module_name}.{attr}")

            module.__getattr__ = _shim  # type: ignore[attr-defined]
            patched.append((module, had_getattr, original_getattr))
        yield
    finally:
        for module, had_getattr, original_getattr in reversed(patched):
            if had_getattr:
                module.__getattr__ = original_getattr  # type: ignore[attr-defined]
            else:
                try:
                    delattr(module, "__getattr__")
                except AttributeError:
                    pass


def _binding_param_name(kind: str, target: str) -> str:
    safe_target = re.sub(r"[^a-zA-Z0-9_]+", "_", target.strip("_")) or "value"
    return f"bind_{kind}_{safe_target}"


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


def _auto_configfiles(workflow_dir: str) -> List[Path]:
    base = Path(workflow_dir)
    candidates = [
        base / "config.yaml",
        base / "config.yml",
        base / "config" / "config.yaml",
        base / "config" / "config.yml",
    ]
    return [path for path in candidates if path.exists()]


def _detect_workdir(workflow_dir: str) -> Path:
    base = Path(workflow_dir).resolve()
    if base.name == "workflow":
        parent = base.parent
        if any(
            path.exists()
            for path in (
                parent / "config",
                parent / "profiles",
                parent / "resources",
                parent / "results",
            )
        ):
            return parent
    return base


def _candidate_configfiles(workflow_dir: str, workdir: Path) -> List[Path]:
    seen: set[Path] = set()
    configfiles: List[Path] = []
    for base in (workdir, Path(workflow_dir).resolve()):
        for path in _auto_configfiles(str(base)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            configfiles.append(path)
    return configfiles


def _load_fallback_config(workflow_dir: str) -> Dict[str, Any]:
    workdir = _detect_workdir(workflow_dir)
    for path in _candidate_configfiles(workflow_dir, workdir):
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            return data
    return {}


def _make_issue(severity: str, code: str, message: str, rule: Optional[str] = None) -> Dict[str, Any]:
    issue: Dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if rule:
        issue["rule"] = rule
    return issue


def _iter_named_items(items: Any) -> List[Tuple[Optional[str], Any]]:
    all_items = getattr(items, "_allitems", None)
    if callable(all_items):
        return list(all_items())

    out: List[Tuple[Optional[str], Any]] = []
    named = list(getattr(items, "items", lambda: [])())
    named_values = [value for _, value in named]
    out.extend((name, value) for name, value in named)
    for value in items:
        if value in named_values:
            continue
        out.append((None, value))
    return out


def _rule_block_from_source(rule: Any) -> str:
    snakefile = getattr(rule, "snakefile", None)
    lineno = getattr(rule, "lineno", None)
    if not snakefile or lineno is None:
        return ""

    source_path = Path(str(snakefile))
    if not source_path.exists():
        return ""

    text = source_path.read_text(encoding="utf-8")
    target_match = None
    for match in RULE_RE.finditer(text):
        start_line = text.count("\n", 0, match.start()) + 1
        if start_line <= lineno:
            target_match = match
        if start_line >= lineno and target_match is not None:
            break
    if target_match is None:
        return ""

    next_match = RULE_RE.search(text, target_match.end())
    return text[target_match.end() : (next_match.start() if next_match else len(text))]


def _scalar_from_source_section(inline: str, lines: List[str]) -> Optional[str]:
    raw = (inline or "").strip()
    if not raw and lines:
        for line in lines:
            stripped = line.strip()
            if stripped:
                raw = stripped
                break
    if not raw:
        return None
    return raw.strip().strip(",").strip('"').strip("'") or None


def _command_from_source_section(inline: str, lines: List[str], keep_shell_comments: bool) -> Optional[str]:
    if not inline and not lines:
        return None

    raw_lines: List[str] = []
    if inline and inline not in {'"""', "'''"}:
        raw_lines.append(inline)
    raw_lines.extend([line.rstrip("\n") for line in lines])

    while raw_lines and raw_lines[0].strip() == "":
        raw_lines.pop(0)
    while raw_lines and raw_lines[-1].strip() == "":
        raw_lines.pop()
    if raw_lines and raw_lines[0].strip() in {'"""', "'''"}:
        raw_lines = raw_lines[1:]
    if raw_lines and raw_lines[-1].strip() in {'"""', "'''"}:
        raw_lines = raw_lines[:-1]
    if not keep_shell_comments:
        raw_lines = [line for line in raw_lines if not re.match(r"^\s*#", line)]

    processed: List[str] = []
    for line in raw_lines:
        line = re.sub(r"\\\s*$", "", line)
        processed.append(line.strip())

    command = " ".join(part for part in processed if part)
    command = re.sub(r"\s+", " ", command).strip()
    return command or None


def _run_code_from_source_section(inline: str, lines: List[str]) -> Optional[str]:
    if not inline and not lines:
        return None

    raw_lines: List[str] = []
    if inline and inline not in {'"""', "'''"}:
        raw_lines.append(inline)
    raw_lines.extend([line.rstrip("\n") for line in lines])
    if raw_lines and raw_lines[0].strip() in {'"""', "'''"}:
        raw_lines = raw_lines[1:]
    if raw_lines and raw_lines[-1].strip() in {'"""', "'''"}:
        raw_lines = raw_lines[:-1]

    run_code = "\n".join(raw_lines).strip()
    return run_code or None


def _normalise_path_value(value: Any) -> Optional[str]:
    if value is None or callable(value):
        return None
    if isinstance(value, os.PathLike):
        return _stringify(os.fspath(value))
    return _stringify(str(value))


def _jsonable_value(value: Any) -> Any:
    try:
        if hasattr(value, "value"):
            value = value.value
    except Exception:
        return None
    if value is None or callable(value):
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def _infer_param_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (dict, list)):
        return "json"
    return "str"


def _callable_label(value: Any) -> str:
    name = getattr(value, "__name__", None)
    if name:
        return name
    return type(value).__name__


def _config_resources(config: Dict[str, Any]) -> Dict[str, ResourceSpec]:
    resources: Dict[str, ResourceSpec] = {}

    def visit(prefix: str, value: Any, leaf_name: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(next_prefix, nested, str(key))
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                visit(f"{prefix}.{index}", nested, f"{leaf_name}[{index}]")
            return
        if not isinstance(value, str):
            return
        if "/" not in value and not value.startswith(("http://", "https://")):
            return

        resource_type = "service" if value.startswith(("http://", "https://")) else "file"
        access = "private" if value.startswith(("/mnt/", "/data/")) else "public"
        resource_id = f"cfg.{prefix}"
        resources[resource_id] = ResourceSpec(
            id=resource_id,
            name=prefix,
            resource_type=resource_type,
            uri=value,
            access=access,
            description=f"Workflow config resource {leaf_name}",
        )

    for key, value in config.items():
        visit(str(key), value, str(key))
    return resources


def _discover_workflow_sources(workflow_dir: str) -> List[Path]:
    base = Path(workflow_dir)
    found: List[Path] = []
    seen: set[Path] = set()
    skip_dirs = {".git", ".snakemake", ".venv", "dist", "build", "__pycache__", ".pytest_cache"}

    snakefile = base / "Snakefile"
    if snakefile.exists():
        found.append(snakefile)
        seen.add(snakefile.resolve())

    for root, dirs, files in os.walk(base):
        dirs[:] = [name for name in dirs if name not in skip_dirs]
        for filename in sorted(files):
            if filename != "Snakefile" and not filename.endswith(".smk"):
                continue
            path = Path(root) / filename
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found


def _compiled_rule_to_toolspec(rule: Any, keep_shell_comments: bool, issues: List[Dict[str, Any]]) -> ToolSpec:
    source_sections = _split_sections(_rule_block_from_source(rule))

    def section(name: str) -> Section:
        return source_sections.get(name, ("", []))

    in_inline, in_lines = section("input")
    out_inline, out_lines = section("output")
    prm_inline, prm_lines = section("params")
    sh_inline, sh_lines = section("shell")
    run_inline, run_lines = section("run")

    input_exprs = _kv_lines(in_inline, in_lines)
    output_exprs = _kv_lines(out_inline, out_lines)
    param_exprs = _kv_lines(prm_inline, prm_lines)

    agent_ready = True
    agent_ready_notes: List[str] = []
    composition_ready = True
    composition_ready_notes: List[str] = []
    generated_binding_params: List[ParamDecl] = []

    def build_io(items: Any, exprs: Dict[str, str], kind: str) -> List[IODecl]:
        nonlocal agent_ready, composition_ready
        declarations: List[IODecl] = []
        for index, (name, value) in enumerate(_iter_named_items(items), start=1):
            decl_name = name or f"_{index}"
            source_expr = exprs.get(decl_name)
            if callable(value):
                label = _callable_label(value)
                source_rendered = _stringify(source_expr) if source_expr else None
                issues.append(
                    _make_issue(
                        "warning",
                        f"dynamic-{kind}",
                        f"Rule '{rule.name}' has a dynamic {kind} callable '{label}'. Explicit agent bindings are required.",
                        rule.name,
                    )
                )
                if kind == "output":
                    agent_ready = False
                    composition_ready = False
                    agent_ready_notes.append(f"dynamic output callable: {label}")
                    composition_ready_notes.append(f"dynamic output callable: {label}")
                    declarations.append(
                        IODecl(
                            name=decl_name,
                            io_type="unknown",
                            path_template=None,
                            description=f"Dynamic {kind} callable: {label}",
                        )
                    )
                    continue

                binding_name = _binding_param_name(kind, decl_name)
                generated_binding_params.append(
                    ParamDecl(
                        name=binding_name,
                        param_type="path" if kind == "input" else "str",
                        required=True,
                        description=(
                            f"Explicit binding for dynamic {kind} '{decl_name}' implemented by callable '{label}'."
                        ),
                        binding_kind=kind,
                        binding_target=decl_name,
                        agent_supplied=True,
                    )
                )
                agent_ready_notes.append(f"requires {binding_name} for dynamic {kind} callable {label}")
                if kind == "input":
                    composition_ready = False
                    composition_ready_notes.append(f"dynamic input callable: {label}")
                declarations.append(
                    IODecl(
                        name=decl_name,
                        io_type="unknown",
                        path_template=None,
                        description=source_rendered or f"Dynamic {kind} callable: {label}",
                    )
                )
                continue

            rendered = _normalise_path_value(value)
            source_rendered = _stringify(source_expr) if source_expr else None
            path_template = rendered
            if source_rendered and (source_rendered.startswith("rules.") or "{config." in source_rendered or path_template is None):
                path_template = source_rendered
            declarations.append(
                IODecl(
                    name=decl_name,
                    io_type=guess_io_type(path_template or rendered or ""),
                    path_template=path_template,
                )
            )
        return declarations

    inputs = build_io(rule.input, input_exprs, "input")
    outputs = build_io(rule.output, output_exprs, "output")

    params: List[ParamDecl] = []
    for index, (name, value) in enumerate(_iter_named_items(rule.params), start=1):
        param_name = name or f"_{index}"
        expr = param_exprs.get(param_name)
        if callable(value):
            label = _callable_label(value)
            issues.append(
                _make_issue(
                    "warning",
                    "dynamic-param",
                    f"Rule '{rule.name}' has a dynamic param callable '{label}'. Explicit agent binding is required.",
                    rule.name,
                )
            )
            agent_ready_notes.append(f"requires {param_name} for dynamic param callable {label}")
            params.append(
                ParamDecl(
                    name=param_name,
                    param_type="str",
                    required=True,
                    description=f"Explicit binding for dynamic param callable '{label}'.",
                    binding_kind="param",
                    binding_target=param_name,
                    agent_supplied=True,
                )
            )
            continue

        default = _jsonable_value(value)
        params.append(
            ParamDecl(
                name=param_name,
                param_type=_infer_param_type(default),
                required=False,
                default=default,
                description=_stringify(expr) if expr else None,
            )
        )

    logs = [
        _normalise_path_value(value)
        for _, value in _iter_named_items(rule.log)
        if _normalise_path_value(value)
    ]

    resources: Dict[str, Any] = {}
    for name, value in getattr(rule, "resources", {}).items():
        if name in {"_cores", "_nodes", "tmpdir"}:
            continue
        rendered = _jsonable_value(value)
        if rendered is None:
            label = _callable_label(value)
            binding_name = _binding_param_name("resource", name)
            issues.append(
                _make_issue(
                    "warning",
                    "dynamic-resource",
                    f"Rule '{rule.name}' resource '{name}' is dynamically evaluated. Agents may override it via '{binding_name}'.",
                    rule.name,
                )
            )
            agent_ready_notes.append(f"optional {binding_name} override for dynamic resource {name}")
            generated_binding_params.append(
                ParamDecl(
                    name=binding_name,
                    param_type="json",
                    required=False,
                    description=f"Optional override for dynamic resource '{name}' implemented by callable '{label}'.",
                    binding_kind="resource",
                    binding_target=name,
                    agent_supplied=True,
                )
            )
            continue
        resources[name] = rendered

    threads = None
    cores = getattr(rule, "resources", {}).get("_cores")
    if cores is not None:
        core_value = _jsonable_value(cores)
        if isinstance(core_value, int):
            threads = core_value

    benchmark = _normalise_path_value(getattr(rule, "benchmark", None))
    conda_env = _normalise_path_value(getattr(rule, "conda_env", None))
    container = _normalise_path_value(getattr(rule, "container_img", None))
    container_engine = None
    if container:
        if container.startswith("docker://"):
            container_engine = "docker"
        elif container.startswith(("shub://", "library://")) or container.endswith(".sif"):
            container_engine = "singularity"

    env_modules = getattr(rule, "env_modules", None)
    envmodules = []
    if env_modules:
        envmodules = [token.strip() for token in str(env_modules).split(",") if token.strip()]

    command = _command_from_source_section(sh_inline, sh_lines, keep_shell_comments) or getattr(rule, "shellcmd", None)
    script = _normalise_path_value(getattr(rule, "script", None))
    wrapper = _normalise_path_value(getattr(rule, "wrapper", None))
    run_code = _run_code_from_source_section(run_inline, run_lines)

    message = getattr(rule, "message", None)
    if isinstance(message, str):
        message = message.strip() or None

    priority = getattr(rule, "priority", None)
    if isinstance(priority, str) and priority.isdigit():
        priority = int(priority)

    cache = getattr(rule, "cache", None) is not None
    if getattr(rule, "is_checkpoint", False):
        composition_ready = False
        agent_ready_notes.append("checkpoint rule")
        composition_ready_notes.append("checkpoint rule")

    if not outputs or not any(output.path_template for output in outputs):
        agent_ready = False
        composition_ready = False
        agent_ready_notes.append("missing concrete output template")
        composition_ready_notes.append("missing concrete output template")

    params.extend(generated_binding_params)

    return ToolSpec(
        id=f"snk.{rule.name}",
        name=rule.name,
        rule=rule.name,
        description=f"Snakemake rule {rule.name}",
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
        group=getattr(rule, "group", None),
        priority=priority if isinstance(priority, int) else None,
        cache=cache,
        agent_ready=agent_ready,
        agent_ready_notes=agent_ready_notes,
        composition_ready=composition_ready,
        composition_ready_notes=composition_ready_notes,
        source_snakefile=str(getattr(rule, "snakefile", "")) or None,
        source_lineno=getattr(rule, "lineno", None),
    )


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
            source_snakefile=path,
        )
        tools[tool.id] = tool

    return tools


def _compile_workflow_with_bootstrap(
    workflow_dir: str,
    workdir: Path,
    configfiles: List[Path],
):
    import snakemake
    from snakemake.api import ConfigSettings, ResourceSettings, SnakemakeApi

    stubbed_imports: List[str] = []
    shimmed_modules: List[str] = []
    snakefile = Path(workflow_dir) / "Snakefile"

    while True:
        try:
            with _stub_workflow_imports(stubbed_imports), _shim_workflow_modules(shimmed_modules):
                with SnakemakeApi() as api:
                    workflow_api = api.workflow(
                        resource_settings=ResourceSettings(),
                        config_settings=ConfigSettings(configfiles=configfiles),
                        snakefile=snakefile,
                        workdir=workdir,
                    )
                    workflow = workflow_api._workflow
                    return workflow, sorted(set(stubbed_imports)), sorted(set(shimmed_modules)), snakemake.__version__
        except ModuleNotFoundError as exc:
            missing = _extract_missing_module(exc)
            if not missing:
                raise
            top_level = missing.split(".", 1)[0]
            if top_level in stubbed_imports or len(stubbed_imports) >= _STUBBABLE_IMPORT_LIMIT:
                raise
            stubbed_imports.append(top_level)
        except AttributeError as exc:
            module_name = _extract_missing_module_attribute(exc)
            if not module_name or module_name in shimmed_modules or len(shimmed_modules) >= _STUBBABLE_IMPORT_LIMIT:
                raise
            shimmed_modules.append(module_name)


def parse_workflow_to_registry(
    workflow_dir: str,
    out_yaml: str,
    keep_shell_comments: bool = True,
    strict: bool = False,
) -> Registry:
    issues: List[Dict[str, Any]] = []
    tools: Dict[str, ToolSpec] = {}
    resources: Dict[str, ResourceSpec] = {}
    parser_mode = "compiled"
    fallback_reason: Optional[str] = None
    workflow_config: Dict[str, Any] = {}
    workdir = _detect_workdir(workflow_dir)
    config_paths = _candidate_configfiles(workflow_dir, workdir)
    configfiles = [str(path) for path in config_paths]
    snakefile = str((Path(workflow_dir) / "Snakefile").resolve())
    snakemake_version = None
    stubbed_imports: List[str] = []
    shimmed_modules: List[str] = []
    bootstrap_mode = "native"

    try:
        workflow, stubbed_imports, shimmed_modules, snakemake_version = _compile_workflow_with_bootstrap(
            workflow_dir=workflow_dir,
            workdir=workdir,
            configfiles=[Path(path) for path in configfiles],
        )
        if stubbed_imports or shimmed_modules:
            bootstrap_mode = "import-stubbed"
            for module in stubbed_imports:
                issues.append(
                    _make_issue(
                        "warning",
                        "workflow-import-stubbed",
                        f"Compiled workflow parsing stubbed missing import '{module}' to keep rule ingestion moving.",
                    )
                )
            for module in shimmed_modules:
                issues.append(
                    _make_issue(
                        "warning",
                        "workflow-module-shimmed",
                        f"Compiled workflow parsing shimmed missing attributes on module '{module}' to keep rule ingestion moving.",
                    )
                )
        workflow_config = dict(getattr(workflow, "config", {}) or {})
        for rule in workflow.rules:
            if rule.name == "all":
                continue
            tool = _compiled_rule_to_toolspec(rule, keep_shell_comments, issues)
            tools[tool.id] = tool
    except Exception as exc:
        parser_mode = "source-fallback"
        bootstrap_mode = "source-fallback"
        fallback_reason = str(exc)
        issues.append(
            _make_issue(
                "warning",
                "compiled-parse-failed",
                f"Compiled Snakemake parsing failed; falling back to source scanning. Reason: {exc}",
            )
        )
        workflow_config = _load_fallback_config(workflow_dir)
        for path in _discover_workflow_sources(workflow_dir):
            tools.update(parse_rules_file(str(path), workflow_config, keep_shell_comments))

    resources.update(_config_resources(workflow_config))

    metadata = {
        "parser": {
            "ingestion_mode": parser_mode,
            "workflow_dir": str(Path(workflow_dir).resolve()),
            "workdir": str(workdir),
            "snakefile": snakefile,
            "configfiles": configfiles,
            "snakemake_version": snakemake_version,
            "bootstrap_mode": bootstrap_mode,
            "stubbed_imports": stubbed_imports,
            "shimmed_modules": shimmed_modules,
            "tool_count": len(tools),
            "agent_ready_rule_count": sum(1 for tool in tools.values() if tool.agent_ready),
            "non_agent_ready_rule_count": sum(1 for tool in tools.values() if not tool.agent_ready),
            "composition_ready_rule_count": sum(1 for tool in tools.values() if tool.composition_ready),
            "non_composition_ready_rule_count": sum(1 for tool in tools.values() if not tool.composition_ready),
            "fallback_reason": fallback_reason,
        },
        "issues": issues,
    }
    reg = Registry(tools=tools, resources=resources, metadata=metadata)

    if strict:
        parser = metadata["parser"]
        if parser["ingestion_mode"] != "compiled":
            raise ValueError("Strict parsing requires compiled Snakemake ingestion; fallback parsing was needed.")
        if parser["non_agent_ready_rule_count"]:
            raise ValueError(
                f"Strict parsing failed: {parser['non_agent_ready_rule_count']} rule(s) are not fully agent-ready. "
                "Inspect registry metadata.issues for details."
            )

    with open(out_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(reg.model_dump(), f, sort_keys=True)
    return reg
