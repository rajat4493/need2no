from pathlib import Path

import typer

from n2n.pipeline import run_highlight, run_pipeline

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


if __name__ == "__main__":
    app()
