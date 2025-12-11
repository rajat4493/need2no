from pathlib import Path

import typer

from n2n import __version__
from n2n.pipeline import redact_file

app = typer.Typer(help="N2N – local-first UK bank statement redactor (v0.0.1)")


@app.command()
def redact(file: str = typer.Argument(..., help="Path to a PDF bank statement")):
    path = Path(file)

    if not path.exists():
        typer.echo(f"[ERROR] File not found: {path}")
        raise typer.Exit(code=1)

    if path.suffix.lower() != ".pdf":
        typer.echo("[ERROR] Only PDF files are supported in v0.1.")
        raise typer.Exit(code=1)

    outcome = redact_file(path)

    if outcome.reason == "quality_too_low":
        typer.echo(
            "[SAFE-SKIP] Text extraction quality too low (< threshold). "
            "No redaction performed."
        )
        raise typer.Exit(code=0)

    if outcome.reason == "no_pii_found":
        typer.echo(
            "[INFO] No PII found with 100% confidence. No changes made."
        )
        raise typer.Exit(code=0)

    typer.echo(
        f"[OK] Redacted {outcome.redactions_applied} PII spans.\n"
        f"     Output: {outcome.output_path}"
    )


@app.command()
def version():
    typer.echo(f"n2n version {__version__}")


if __name__ == "__main__":
    app()
