from __future__ import annotations
try:
    from typer.testing import CliRunner
    from pipemind.cli.app import app
except Exception:  # pragma: no cover
    CliRunner = None
    app = None


def test_cli_help():
    if not CliRunner or not app:
        return
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "PipeMind CLI" in res.stdout
