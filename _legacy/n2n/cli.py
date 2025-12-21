import json
from pathlib import Path
import os

import typer
import uvicorn

from n2n.pipeline import run_highlight, run_pipeline
from n2n.packs import list_packs, run_pack

app = typer.Typer(help="N2N – local-first UK bank statement redactor")


@app.command()
def redact(
    file: str = typer.Argument(..., help="Path to a PDF bank statement"),
    config_dir: str = typer.Option(
        ".",
        "--config-dir",
        "-c",
        help="Base directory containing the config/ folder.",
    ),
):
    input_path = Path(file)
    if not input_path.exists():
        typer.echo(f"[ERROR] File not found: {input_path}")
        raise typer.Exit(code=1)

    if input_path.suffix.lower() != ".pdf":
        typer.echo("[ERROR] Only PDF files are supported.")
        raise typer.Exit(code=1)

    base_dir = Path(config_dir).resolve()

    try:
        outcome = run_pipeline(input_path, base_dir)
    except FileNotFoundError as exc:
        typer.echo(f"[ERROR] {exc}")
        raise typer.Exit(code=2) from exc

    if outcome.reason == "quality_too_low":
        typer.echo(
            "[SAFE-SKIP] Text extraction quality below configured threshold. "
            "No changes were made."
        )
        raise typer.Exit(code=0)

    if outcome.reason == "no_pii_found":
        typer.echo("[INFO] No PII met strict detection criteria. No changes made.")
        raise typer.Exit(code=0)

    typer.echo(
        f"[OK] Redacted {outcome.redactions_applied} PII spans.\n"
        f"     Output: {outcome.output_path}"
    )


@app.command("n2n-highlight")
def highlight(
    file: str = typer.Argument(..., help="Path to a PDF bank statement"),
    config_dir: str = typer.Option(
        ".",
        "--config-dir",
        "-c",
        help="Base directory containing the config/ folder.",
    ),
):
    input_path = Path(file)
    if not input_path.exists():
        typer.echo(f"[ERROR] File not found: {input_path}")
        raise typer.Exit(code=1)

    if input_path.suffix.lower() != ".pdf":
        typer.echo("[ERROR] Only PDF files are supported.")
        raise typer.Exit(code=1)

    base_dir = Path(config_dir).resolve()

    try:
        outcome = run_highlight(input_path, base_dir)
    except FileNotFoundError as exc:
        typer.echo(f"[ERROR] {exc}")
        raise typer.Exit(code=2) from exc

    if outcome.reason == "quality_too_low":
        typer.echo(
            "[SAFE-SKIP] Text extraction quality below configured threshold. "
            "No highlights produced."
        )
        raise typer.Exit(code=0)

    if outcome.reason == "no_pii_found":
        typer.echo("[INFO] No PII met strict detection criteria. No highlights added.")
        raise typer.Exit(code=0)

    typer.echo(
        f"[OK] Highlighted {outcome.redactions_applied} PII spans.\n"
        f"     Output: {outcome.output_path}"
    )


@app.command("n2n-serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host interface for the API server."),
    port: int = typer.Option(8000, "--port", help="Port for the API server."),
    config_dir: str = typer.Option(
        ".",
        "--config-dir",
        "-c",
        help="Base directory containing the config/ folder.",
    ),
):
    """
    Start the FastAPI server that exposes /redact and /highlight endpoints.
    """

    base_dir = Path(config_dir).resolve()
    os.environ["N2N_CONFIG_DIR"] = str(base_dir)

    uvicorn.run("n2n.api_server:app", host=host, port=port, reload=False)


# Helper to resolve pack selection
def _resolve_pack(pack: str | None, country: str | None, doc: str | None) -> str:
    if pack:
        return pack
    if country or doc:
        combo = f"{country or '?'} / {doc or '?'}"
        raise typer.BadParameter(
            f"No pack mapping exists for country/doc={combo}. "
            "Specify --pack from the supported list."
        )

    available = ", ".join(list_packs()) or "none"
    raise typer.BadParameter(
        f"Missing --pack. Available pack IDs: {available}. "
        "Alternatively provide --country and --doc when mappings exist."
    )


@app.command("process")
def process_pack(
    file: str = typer.Argument(..., help="Path to a PDF to process with a pack."),
    pack: str = typer.Option(None, "--pack", "-p", help="Pack to execute."),
    country: str = typer.Option(None, "--country", help="Country code to auto-select a pack."),
    doc: str = typer.Option(None, "--doc", help="Document type identifier."),
    config_dir: str = typer.Option(
        ".",
        "--config-dir",
        "-c",
        help="Base directory containing the config/ folder.",
    ),
):
    input_path = Path(file)
    if not input_path.exists():
        typer.echo(f"[ERROR] File not found: {input_path}")
        raise typer.Exit(code=1)

    if input_path.suffix.lower() != ".pdf":
        typer.echo("[ERROR] Only PDF files are supported.")
        raise typer.Exit(code=1)

    base_dir = Path(config_dir).resolve()

    pack_id = _resolve_pack(pack, country, doc)

    try:
        report = run_pack(pack_id, input_path, base_dir)
    except ValueError as exc:
        typer.echo(f"[ERROR] {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(report.to_dict(), indent=2))


@app.command("packs")
def show_packs():
    available = list_packs()
    if not available:
        typer.echo("No packs registered.")
        return
    for pack_id in available:
        typer.echo(pack_id)


if __name__ == "__main__":
    app()
