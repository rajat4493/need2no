from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from n2n.packs import list_packs, run_pack

app = FastAPI(title="N2N v0.1")


@app.get("/packs")
def get_packs():
    return sorted(list_packs().keys())


@app.post("/v1/process")
async def process(
    file: UploadFile = File(...),
    pack_id: str = Form(...),
    outdir: str = Form("out"),
    force_band_redact: bool = Form(False),
    ocr_backend: str | None = Form(None),
):
    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        outdir_path = Path(outdir)
        report = run_pack(
            pack_id,
            tmp_path,
            outdir_path,
            force_band_redact=force_band_redact,
            ocr_backend=ocr_backend,
        )
        return report
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


__all__ = ["app"]
