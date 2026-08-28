"""End-to-end coverage for the MCP server (n2n/mcp_server.py), spawning
the real `n2n mcp` subprocess and speaking actual JSON-RPC over stdio —
the real integration surface an agent client uses, not just calling the
underlying Python functions directly.

This is exactly how a real bug was found while building this: PyMuPDF's
deprecated `import fitz` compatibility shim prints a warning straight to
stdout at import time, which corrupts the JSON-RPC stream (stdio MCP
transport requires stdout to carry nothing but protocol messages) and
broke the very first real client handshake. Fixed by switching every
`import fitz` to `import pymupdf as fitz` throughout the codebase — the
non-deprecated import doesn't print anything. test_stdout_is_clean_json_rpc_only
below guards against that regressing.

The MCP client session is opened directly inside each test body (not via
a pytest fixture) — anyio's stdio_client enforces that its cancel scope
is entered and exited in the same asyncio task, which a separate
async-generator fixture's setup/teardown split doesn't reliably preserve
under pytest-asyncio.
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PACK_ID = "uk.bank_statement.share_with_ai"


@asynccontextmanager
async def open_session():
    params = StdioServerParameters(command=sys.executable, args=["-m", "n2n.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def test_lists_expected_tools():
    async with open_session() as session:
        tools = await session.list_tools()
        names = {t.name for t in tools.tools}
        assert names == {"list_packs", "redact_document"}


async def test_list_packs_returns_known_packs():
    async with open_session() as session:
        result = await session.call_tool("list_packs", {})
        assert result.isError is False
        text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
        assert "uk.bank_statement.share_with_ai" in text
        assert "pci.card_data.share_with_ai" in text


async def test_redact_document_pass_auto(clean_statement_pdf, tmp_path):
    out_path = tmp_path / "mcp_out.pdf"
    manifest_path = tmp_path / "mcp_out.n2n.json"
    async with open_session() as session:
        result = await session.call_tool(
            "redact_document",
            {
                "input_path": str(clean_statement_pdf),
                "pack_id": PACK_ID,
                "output_path": str(out_path),
                "manifest_path": str(manifest_path),
            },
        )
    assert result.isError is False
    body = json.loads(result.content[0].text)
    assert body["status"] == "PASS_AUTO"
    assert out_path.exists()
    assert manifest_path.exists()
    # No raw sensitive text in the tool response, same guarantee as the
    # HTTP API and CLI.
    for finding in body["findings"]:
        assert "text" not in finding


async def test_redact_document_needs_review_writes_nothing(statement_with_name_pdf, tmp_path):
    out_path = tmp_path / "mcp_out.pdf"
    manifest_path = tmp_path / "mcp_out.n2n.json"
    async with open_session() as session:
        result = await session.call_tool(
            "redact_document",
            {
                "input_path": str(statement_with_name_pdf),
                "pack_id": PACK_ID,
                "output_path": str(out_path),
                "manifest_path": str(manifest_path),
            },
        )
    body = json.loads(result.content[0].text)
    assert body["status"] == "NEEDS_REVIEW"
    assert not out_path.exists()


async def test_redact_document_defaults_output_paths_next_to_input(clean_statement_pdf):
    async with open_session() as session:
        result = await session.call_tool(
            "redact_document", {"input_path": str(clean_statement_pdf), "pack_id": PACK_ID}
        )
    body = json.loads(result.content[0].text)
    assert body["status"] == "PASS_AUTO"
    expected_out = clean_statement_pdf.with_name(clean_statement_pdf.stem + ".safe.pdf")
    assert Path(body["output_path"]) == expected_out
    assert expected_out.exists()


async def test_redact_document_dry_run_writes_nothing(clean_statement_pdf):
    async with open_session() as session:
        result = await session.call_tool(
            "redact_document",
            {"input_path": str(clean_statement_pdf), "pack_id": PACK_ID, "dry_run": True},
        )
    body = json.loads(result.content[0].text)
    assert body["status"] != "PASS_AUTO"
    default_out = clean_statement_pdf.with_name(clean_statement_pdf.stem + ".safe.pdf")
    assert not default_out.exists()


async def test_redact_document_unknown_pack_returns_clean_error_not_a_crash(clean_statement_pdf):
    async with open_session() as session:
        result = await session.call_tool(
            "redact_document",
            {"input_path": str(clean_statement_pdf), "pack_id": "not.a.real.pack"},
        )
    assert result.isError is False  # a handled error, not a protocol-level failure
    body = json.loads(result.content[0].text)
    assert body["status"] == "PROCESSING_ERROR"
    assert "not.a.real.pack" in body["reasons"][0]


async def test_redact_document_missing_file_returns_clean_error(tmp_path):
    async with open_session() as session:
        result = await session.call_tool(
            "redact_document",
            {"input_path": str(tmp_path / "does_not_exist.pdf"), "pack_id": PACK_ID},
        )
    body = json.loads(result.content[0].text)
    assert body["status"] == "PROCESSING_ERROR"


def test_stdout_is_clean_json_rpc_only():
    """Guards against the exact bug this test file's docstring describes:
    any stray print at import time (a deprecation warning, a library's
    own banner, etc.) corrupts the stdio JSON-RPC stream for a real MCP
    client. Runs python -m n2n.mcp_server import-only in a subprocess and
    asserts stdout is completely empty before any protocol traffic."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", "import n2n.mcp_server"],
        capture_output=True,
        timeout=10,
    )
    assert proc.stdout == b"", f"unexpected stdout pollution: {proc.stdout!r}"
