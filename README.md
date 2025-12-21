# N2N Redactor

N2N is a local-first redaction toolkit that inspects PDFs or photos for payment-card and identity documents. Packs encapsulate deterministic pipelines for specific document types:

- `global.pci_lite.v1` inspects PDFs for PANs and produces redacted outputs.
- `global.card_photo.v1` handles card photos with ROI OCR and visual heuristics.
- `global.id_photo.v1` handles ID cards, MRZ, ID number, and face/DOB masking.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

For contributors, run `scripts/dev_reset.sh` to ensure the editable install points at this checkout.

## CLI Usage

List packs:

```bash
python -m n2n.cli packs
```

Process an input (PDF or image):

```bash
python -m n2n.cli process input.pdf --pack global.pci_lite.v1 --outdir out
```

The command writes a JSON report plus highlight PDF to `out`. For CONFIRMED card/id packs, a redacted PDF is also emitted.

### OCR backends

Photo packs now use a pluggable OCR stack:

- Default `tesseract` backend works everywhere.
- Optional `apple` backend wraps a tiny Swift CLI that uses `Vision` on macOS. Build once via:

  ```bash
  cd tools/apple_vision_ocr
  swift build -c release
  cd ../..
  ```

- Optional extras:
  - `pip install -e .[paddle]` enables PaddleOCR
  - `pip install -e .[easy]` enables EasyOCR

Select backends with `--ocr-backend auto|tesseract|apple|paddle|easy|combo`, via the `N2N_OCR_BACKEND` env var, or `ocr_backend=` on the API. In AUTO/COMBO modes the runner tries Apple Vision first, then PaddleOCR, then Tesseract; if any backend is missing it skips to the next without failing the run.

## API Usage

`n2n.api_server:app` exposes a FastAPI server. Example with uvicorn:

```bash
uvicorn n2n.api_server:app --reload
```

`POST /v1/process` accepts `multipart/form-data` uploads with `file`, `pack_id`, and optional `outdir`.

## YOLO Weights

Object detection weights (`n2n_assets/models/card_id_yolo.pt`) are optional. When absent, packs fall back to deterministic geometry and visual heuristics. In both cases the system remains local-first and never uploads documents.
