from __future__ import annotations

import json
from pathlib import Path

import typer

from n2n import pipeline
from n2n.packs.registry import list_packs

app = typer.Typer(help="N2N — a fail-closed disclosure gate.")


@app.command()
def packs() -> None:
    """List available purpose packs."""
    for pack_id, pack in list_packs().items():
        typer.echo(f"{pack_id}\t{pack.version}\t{pack.description}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
) -> None:
    """Run the local web UI: upload -> instant result -> findings -> download.

    Runs entirely on this machine — no document content or telemetry
    leaves the process."""
    import uvicorn

    typer.echo(f"N2N running at http://{host}:{port} (local only, nothing leaves this process)")
    uvicorn.run("n2n.webapp.server:app", host=host, port=port)


@app.command()
def redact(
    file: Path = typer.Argument(..., exists=True, readable=True, help="PDF to process"),
    pack: str = typer.Option(..., "--pack", help="Purpose pack ID"),
    output: Path = typer.Option(Path("safe.pdf"), "--output", help="Output PDF path"),
    manifest: Path = typer.Option(Path("safe.n2n.json"), "--manifest", help="Evidence manifest path"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would happen without producing output."
    ),
) -> None:
    """Certify a document for release under a purpose pack, or refuse with a reason."""
    report = pipeline.run(
        input_path=file,
        pack_id=pack,
        output_path=None if dry_run else output,
        manifest_path=None if dry_run else manifest,
        dry_run=dry_run,
    )

    _print_human_summary(report)
    if report.status != "PASS_AUTO":
        raise typer.Exit(code=1)


def _print_human_summary(report) -> None:
    typer.echo(f"Status: {report.status}")
    for reason in report.reasons:
        typer.echo(f"  - {reason}")

    if report.status == "PASS_AUTO":
        typer.echo(f"Certified output: {report.output_path}")
        typer.echo(f"Evidence manifest: {report.manifest_path}")
    else:
        typer.echo("Not certified: no output file was produced.")

    if report.findings:
        typer.echo("Findings:")
        for f in report.findings:
            typer.echo(
                f"  page {f.page + 1}: {f.field_id} [{f.tier}] -> {f.action}"
            )

    typer.echo(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    app()
