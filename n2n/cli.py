from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from n2n import pipeline
from n2n.auth import store as api_key_store
from n2n.packs.registry import list_packs

app = typer.Typer(help="N2N — a fail-closed disclosure gate.")
apikey_app = typer.Typer(help="Manage API keys for n2n serve.")
app.add_typer(apikey_app, name="apikey")


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
    leaves the process. Every request requires an API key; if none exist
    yet, one is created automatically so there's still a one-command
    happy path."""
    import uvicorn

    if api_key_store.is_empty():
        plaintext, record = api_key_store.create("bootstrap")
        typer.echo("No API keys found — created one automatically.")
        typer.echo(f"API key (shown once, save it now): {plaintext}")
        typer.echo("Paste it into the web UI when prompted, or use it as a Bearer token.\n")

    typer.echo(f"N2N running at http://{host}:{port} (local only, nothing leaves this process)")
    uvicorn.run("n2n.webapp.server:app", host=host, port=port)


@app.command()
def mcp() -> None:
    """Run the MCP server (stdio transport) so an AI agent can call N2N
    as a tool: list_packs, redact_document. Same trust model as the CLI —
    no API key, since the calling agent already has local file access by
    virtue of being able to spawn this process."""
    from n2n.mcp_server import main as mcp_main

    mcp_main()


@apikey_app.command("create")
def apikey_create(name: str = typer.Option(..., "--name", help="A label for this key")) -> None:
    """Create a new API key. The plaintext is shown exactly once."""
    plaintext, record = api_key_store.create(name)
    typer.echo(f"Created key '{record.name}' (id: {record.id})")
    typer.echo(f"API key (shown once, save it now): {plaintext}")


@apikey_app.command("list")
def apikey_list() -> None:
    """List API keys (never shows the plaintext key or its hash)."""
    records = api_key_store.list()
    if not records:
        typer.echo("No API keys.")
        return
    for r in records:
        created = datetime.fromtimestamp(r.created_at, tz=timezone.utc).isoformat()
        last_used = (
            datetime.fromtimestamp(r.last_used_at, tz=timezone.utc).isoformat()
            if r.last_used_at
            else "never"
        )
        status = "revoked" if r.revoked else "active"
        typer.echo(f"{r.id}\t{r.name}\t{status}\tcreated={created}\tlast_used={last_used}")


@apikey_app.command("revoke")
def apikey_revoke(key_id: str = typer.Argument(..., help="Key id, from `n2n apikey list`")) -> None:
    """Revoke an API key. Revocation is permanent — create a new key if needed."""
    if api_key_store.revoke(key_id):
        typer.echo(f"Revoked key {key_id}.")
    else:
        typer.echo(f"No key found with id {key_id}.")
        raise typer.Exit(code=1)


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
