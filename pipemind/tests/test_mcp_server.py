from __future__ import annotations

from pipemind.mcp_server.server import make_tool_callable
from pipemind.registry.schema import ToolSpec, IODecl


class DummyRunResult:
    def __init__(self):
        self.returncode = 0
        self.stdout = "ok"
        self.stderr = ""


def test_make_tool_callable_requires_output_disambiguation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_run(*args, **kwargs):
        return DummyRunResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    tool = ToolSpec(
        id="snk.multi",
        name="multi",
        rule="multi",
        inputs=[],
        outputs=[
            IODecl(name="a", io_type="txt", path_template="out/{sample}.a.txt"),
            IODecl(name="b", io_type="txt", path_template="out/{sample}.b.txt"),
        ],
    )

    call = make_tool_callable(tool)
    result = call(sample="S1")

    assert result["status"] == "error"
    assert "output_name or target" in result["error"]


def test_make_tool_callable_honors_output_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_run(*args, **kwargs):
        return DummyRunResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    tool = ToolSpec(
        id="snk.multi",
        name="multi",
        rule="multi",
        inputs=[],
        outputs=[
            IODecl(name="a", io_type="txt", path_template="out/{sample}.a.txt"),
            IODecl(name="b", io_type="txt", path_template="out/{sample}.b.txt"),
        ],
    )

    call = make_tool_callable(tool)
    result = call(sample="S1", output_name="b")

    assert result["target"].endswith("out/S1.b.txt")
    assert result["returncode"] == 0
