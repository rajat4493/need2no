from __future__ import annotations

import json
from pathlib import Path

import typer

from n2n.packs import list_packs, run_pack

app = typer.Typer(help="N2N v0.1 – card PAN redaction")


@app.command()
def packs():
    for pack_id in list_packs().keys():
        typer.echo(pack_id)


@app.command()
def process(
    file: Path = typer.Argument(..., exists=True, readable=True, help="PDF or image to process"),
    pack: str = typer.Option(..., "--pack", help="Pack ID to execute"),
    outdir: Path = typer.Option(Path("out"), "--outdir", help="Output directory"),
    force_band_redact: bool = typer.Option(
        False,
        "--force-band-redact",
        help="Force redaction of suggested PAN bands when review is required.",
    ),
    ocr_backend: str | None = typer.Option(
        None,
        "--ocr-backend",
        help="OCR backend mode: auto, tesseract, apple, paddle, easy, combo.",
    ),
):
    report = run_pack(
        pack,
        file,
        outdir,
        force_band_redact=force_band_redact,
        ocr_backend=ocr_backend,
    )
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
