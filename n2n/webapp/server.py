"""Customer-facing layer: upload -> instant result -> findings -> download.

Wraps n2n.pipeline directly — no separate service, no network dependency,
matches the "nothing leaves the process" positioning documented in the
build spec (section 6). Intended to be run locally (n2n serve) or embedded
in a customer's own pipeline via the same API.

Every /v1/* endpoint requires an API key (Authorization: Bearer <key>) —
see n2n/auth.py. The same key mechanism serves both the manual web UI
(the browser prompts for one and stores it in localStorage) and
automation/API clients, so there's one auth model, not two.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from n2n import pipeline
from n2n.auth import ApiKeyRecord, store as api_key_store
from n2n.packs.registry import get_pack, list_packs
from n2n.webapp.ratelimit import limiter
from n2n.webapp.sessions import store as session_store

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB — a defensive cap, not a product limit
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="N2N", description="A fail-closed disclosure gate.")


def require_api_key(authorization: str | None = Header(default=None)) -> ApiKeyRecord:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header. Expected: Bearer <api-key>.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization.removeprefix("Bearer ").strip()
    record = api_key_store.verify(presented)
    if record is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not limiter.check(record.id):
        retry_after = limiter.retry_after_seconds(record.id)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    return record


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/v1/packs")
def packs(_: ApiKeyRecord = Depends(require_api_key)) -> list[dict]:
    return [
        {"pack_id": pack.pack_id, "version": pack.version, "description": pack.description}
        for pack in list_packs().values()
    ]


@app.post("/v1/redact")
async def redact(
    file: UploadFile, pack_id: str, _: ApiKeyRecord = Depends(require_api_key)
) -> dict:
    try:
        get_pack(pack_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 25 MB upload limit.")
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file.")

    work_dir = session_store.new_work_dir()
    input_path = work_dir / "input.pdf"
    input_path.write_bytes(contents)

    output_path = work_dir / "safe.pdf"
    manifest_path = work_dir / "safe.n2n.json"

    report = pipeline.run(
        input_path=input_path,
        pack_id=pack_id,
        output_path=output_path,
        manifest_path=manifest_path,
    )

    body = report.to_dict()
    body["download_token"] = None

    if report.status == "PASS_AUTO":
        token = session_store.create(output_path, manifest_path)
        body["download_token"] = token
    else:
        # Nothing releasable — the per-request temp dir (which never held a
        # certified output on this path) is discarded immediately rather
        # than left for the session TTL to clean up.
        shutil.rmtree(work_dir, ignore_errors=True)

    return body


@app.get("/v1/download/{token}/output.pdf")
def download_output(token: str, _: ApiKeyRecord = Depends(require_api_key)) -> FileResponse:
    session = session_store.get(token)
    if session is None or not session.output_path.exists():
        raise HTTPException(status_code=404, detail="No certified output for this token.")
    return FileResponse(session.output_path, media_type="application/pdf", filename="safe.pdf")


@app.get("/v1/download/{token}/manifest.json")
def download_manifest(token: str, _: ApiKeyRecord = Depends(require_api_key)) -> FileResponse:
    session = session_store.get(token)
    if session is None or not session.manifest_path.exists():
        raise HTTPException(status_code=404, detail="No manifest for this token.")
    return FileResponse(
        session.manifest_path, media_type="application/json", filename="safe.n2n.json"
    )
