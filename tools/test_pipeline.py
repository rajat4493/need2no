from __future__ import annotations

import argparse
from pathlib import Path

from n2n.detectors.bank_statement_uk import detect_pii_uk_bank_statement
from n2n.extractors.pdf_text import extract_text_with_quality
from n2n.pipeline import run_highlight, run_pipeline
from n2n.utils.config_loader import load_global_config, load_profile_config


def _print_detections(detections) -> None:
    if not detections:
        print("No detections found.")
        return

    print("Detections:")
    for det in detections:
        print(
            f"- field={det.field_id} primitive={det.primitive} page={det.span.page_index} "
            f"text={det.span.text!r} context={det.context!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the N2N pipeline end-to-end")
    parser.add_argument("file", help="Path to the PDF file to process")
    parser.add_argument(
        "--mode",
        choices=["highlight", "redact"],
        default="highlight",
        help="Whether to run the redaction or highlight output",
    )
    parser.add_argument(
        "--config-dir",
        default=".",
        help="Base directory containing the config/ folder",
    )
    args = parser.parse_args()

    pdf_path = Path(args.file).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    base_dir = Path(args.config_dir).expanduser().resolve()

    defaults = load_global_config(base_dir)
    profile = load_profile_config(base_dir, defaults["country_pack"], defaults["profile"])

    extraction = extract_text_with_quality(pdf_path)
    print(f"Quality score: {extraction.quality_score:.2f}")

    detections = detect_pii_uk_bank_statement(extraction, profile)
    strict = [d for d in detections if d.confidence >= 1.0]
    _print_detections(strict)

    if not strict:
        print("No strict detections – skipping output run.")
        return

    if args.mode == "highlight":
        outcome = run_highlight(pdf_path, base_dir)
    else:
        outcome = run_pipeline(pdf_path, base_dir)

    if outcome.reason:
        print(f"Pipeline exited early: {outcome.reason}")
    else:
        print(f"Output file: {outcome.output_path}")


if __name__ == "__main__":
    main()
