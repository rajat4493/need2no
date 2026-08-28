"""MCP server exposing N2N's pipeline as tools for AI agents.

Runs over stdio by default — the standard local-process MCP transport,
and the same trust model the CLI already has: if the calling agent can
spawn this process, it already has the file access it's asking N2N to
use. This is NOT the same trust boundary as `n2n serve`'s HTTP API and
does not use its API-key auth (n2n/auth.py) — a network-exposed
(HTTP/SSE) MCP deployment would need that same auth layer, and doesn't
have it here. See README's security-layer section.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from n2n import pipeline
from n2n.packs.registry import get_pack, list_packs as _list_packs

mcp = FastMCP(
    "n2n",
    instructions=(
        "N2N is a fail-closed disclosure gate for sensitive documents: it "
        "certifies a document as safe to release under a declared purpose "
        "pack, with signed evidence of what it did, or refuses and says "
        "why. Call list_packs to see available purpose packs, then "
        "redact_document with a local file path to certify a PDF, or get "
        "refused with a plain-language reason. A refusal never produces an "
        "output file — only a PASS_AUTO result does. Findings returned "
        "never include the underlying sensitive text, only field type, "
        "page, geometry, confidence tier, and what happened to it."
    ),
)


@mcp.tool()
def list_packs() -> list[dict]:
    """List available purpose packs (pack_id, version, description)."""
    return [
        {"pack_id": p.pack_id, "version": p.version, "description": p.description}
        for p in _list_packs().values()
    ]


@mcp.tool()
def redact_document(
    input_path: str,
    pack_id: str,
    output_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Certify a local PDF for release under a purpose pack, or refuse
    with a plain-language reason.

    Only ever writes an output file (and its signed evidence manifest)
    when the result is PASS_AUTO — a refusal produces no file, by design.
    If output_path/manifest_path aren't given, they default to
    "<input>.safe.pdf" / "<input>.safe.n2n.json" next to the input file.
    Set dry_run=true to see what would happen without writing anything.
    """
    try:
        get_pack(pack_id)
    except ValueError as exc:
        return {
            "status": "PROCESSING_ERROR",
            "pack_id": pack_id,
            "reasons": [str(exc)],
            "findings": [],
            "download_token": None,
        }

    in_path = Path(input_path).expanduser().resolve()
    if not in_path.exists():
        return {
            "status": "PROCESSING_ERROR",
            "pack_id": pack_id,
            "reasons": [f"File not found: {input_path}"],
            "findings": [],
        }

    resolved_output = Path(output_path).expanduser().resolve() if output_path else in_path.with_name(
        in_path.stem + ".safe.pdf"
    )
    resolved_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path
        else in_path.with_name(in_path.stem + ".safe.n2n.json")
    )

    report = pipeline.run(
        input_path=in_path,
        pack_id=pack_id,
        output_path=None if dry_run else resolved_output,
        manifest_path=None if dry_run else resolved_manifest,
        dry_run=dry_run,
    )
    return report.to_dict()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
