from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from n2n.pipeline import run_highlight, run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="N2N Local API", version="0.0.1")


def _save_upload_to_temp(upload: UploadFile, tmp_dir: Path) -> Path:
    tmp_path = tmp_dir / upload.filename
    with tmp_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return tmp_path


def _run_engine(mode: Literal["redact", "highlight"], pdf_path: Path, config_dir: Path):
    if mode == "highlight":
        return run_highlight(pdf_path, config_dir)
    return run_pipeline(pdf_path, config_dir)


def _build_pdf_response(path: Path) -> Response:
    data = path.read_bytes()
    headers = {"Content-Disposition": f'attachment; filename="{path.name}"'}
    return Response(content=data, media_type="application/pdf", headers=headers)


def _process_upload(upload: UploadFile, mode: Literal["redact", "highlight"]):
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        input_path = _save_upload_to_temp(upload, tmp_dir)
        outcome = _run_engine(mode, input_path, PROJECT_ROOT)

        if outcome.reason:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "skipped",
                    "reason": outcome.reason,
                    "redactions_applied": outcome.redactions_applied,
                },
            )

        if not outcome.output_path or not outcome.output_path.exists():
            raise HTTPException(status_code=500, detail="Output file missing.")

        return _build_pdf_response(outcome.output_path)


@app.post("/redact")
async def redact(file: UploadFile = File(...)):
    return _process_upload(file, "redact")


@app.post("/highlight")
async def highlight(file: UploadFile = File(...)):
    return _process_upload(file, "highlight")


__all__ = ["app"]
