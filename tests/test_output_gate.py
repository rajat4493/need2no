import ast
from pathlib import Path

import pytest

from n2n.output_gate import ReleaseToken, mint_release_token, write_certified_output

N2N_ROOT = Path(__file__).resolve().parent.parent / "n2n"


def test_write_requires_a_token(tmp_path):
    with pytest.raises(TypeError):
        write_certified_output(  # type: ignore[call-arg]
            output_payload=b"data",
            output_path=tmp_path / "out.pdf",
            manifest_payload=b"{}",
            manifest_path=tmp_path / "out.json",
        )


def test_write_rejects_forged_token(tmp_path):
    class FakeToken:
        pass

    with pytest.raises(PermissionError):
        write_certified_output(
            token=FakeToken(),
            output_payload=b"data",
            output_path=tmp_path / "out.pdf",
            manifest_payload=b"{}",
            manifest_path=tmp_path / "out.json",
        )
    assert not (tmp_path / "out.pdf").exists()


def test_write_rejects_token_minted_for_different_payload(tmp_path):
    token = mint_release_token(b"payload-a")
    with pytest.raises(PermissionError):
        write_certified_output(
            token=token,
            output_payload=b"payload-b",
            output_path=tmp_path / "out.pdf",
            manifest_payload=b"{}",
            manifest_path=tmp_path / "out.json",
        )
    assert not (tmp_path / "out.pdf").exists()


def test_write_succeeds_with_a_genuine_matching_token(tmp_path):
    payload = b"certified bytes"
    token = mint_release_token(payload)
    out = tmp_path / "out.pdf"
    manifest = tmp_path / "out.json"
    write_certified_output(
        token=token,
        output_payload=payload,
        output_path=out,
        manifest_payload=b"{}",
        manifest_path=manifest,
    )
    assert out.read_bytes() == payload


def test_release_token_cannot_be_constructed_without_the_mint_function():
    with pytest.raises(TypeError):
        ReleaseToken()  # type: ignore[call-arg]


def test_mint_release_token_is_only_called_from_pipeline_module():
    """Static check: the only place in the codebase allowed to mint a
    release token is n2n/pipeline.py, and only there is the output
    physically writable. This guards against a future edit routing
    around the PASS_AUTO gate."""
    callers = []
    for path in N2N_ROOT.rglob("*.py"):
        if path.name == "output_gate.py":
            continue
        source = path.read_text()
        if "mint_release_token" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name == "mint_release_token":
                    callers.append(path)

    assert callers == [N2N_ROOT / "pipeline.py"], (
        f"mint_release_token must only be called from pipeline.py, found: {callers}"
    )
