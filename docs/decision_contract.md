# Decision Contract

This document captures the contract for how the N2N pipeline communicates its decision back to clients, including the states a run can end in, the refusal reason codes emitted in "safe skip" scenarios, and how on-disk artifacts are named.

## Decision states

| State | Description | Surface |
| --- | --- | --- |
| `processed` | The pipeline completed successfully, produced an output artifact and set `reason=None` on the `RedactionOutcome` dataclass. The CLI reports `[OK]` and the API returns the PDF as a binary response. | `n2n.models.RedactionOutcome` in `n2n/models.py` defines `reason` as optional (lines 39-44) and `n2n/pipeline.py` returns a populated `output_path` when no reason is set (lines 103-129, 131-149). |
| `skipped` | The run was safely refused. The pipeline indicates this by setting `reason` on the `RedactionOutcome` and leaving `output_path=None`. `n2n/api_server.py` surfaces this as `{"status": "skipped", "reason": ...}` and the CLI prints a SAFE-SKIP/INFO message before exiting successfully. | `n2n/api_server.py` lines 43-56; `n2n/cli.py` lines 31-49 and 73-91. |
| `error` | Situations where the server rejects the request before the pipeline decision (e.g., a non-PDF upload) or when an unexpected internal issue occurs (such as a missing output file) raise HTTP errors and no decision payload is produced. | `n2n/api_server.py` lines 40-41 and 58-60. |

All downstream consumers should treat `processed` as an approval, `skipped` as a refusal that uses one of the codes below, and `error` as an operational issue that warrants alerting/retry.

## Refusal reason code enum

When a run ends in the `skipped` state, the `reason` field is one of the following symbolic codes:

| Code | Description | Emitted by |
| --- | --- | --- |
| `quality_too_low` | Deterministic text extraction produced a quality score below the configured threshold, so we skip instead of producing a deceptive artifact. | `_extract_with_mode` returning early when `extractor_mode="text"` fails (lines 57-62) and `_prepare_detections` propagating the code in `n2n/pipeline.py` (lines 81-99). |
| `ocr_quality_too_low` | OCR extraction was required but could not meet the threshold. This can happen when the operator forces OCR mode or when auto mode falls back to OCR and still fails. | `_extract_with_mode` for the OCR path (lines 63-78). |
| `no_pii_found` | Extraction succeeded but no detections survived the strict confidence or precision filters. To avoid producing an unchanged file with no explanation, we refuse with this code. | `_prepare_detections` once it sees zero strict detections (lines 91-99). |

The CLI currently displays specialized messaging for `quality_too_low` and `no_pii_found`. Until `ocr_quality_too_low` gets its own UX, treat it the same as `quality_too_low` in API clients.

## Artifact naming

The artifact naming scheme keeps the original filename stem and appends a deterministic suffix before the `.pdf` extension so that downstream systems can predict paths without parsing logs.

### Redacted output

* `n2n/pipeline.py` builds the redacted artifact by appending the suffix returned by `_get_output_suffix` (lines 41-45, 103-129).
* The suffix is configurable via `config/n2n.defaults.yaml` under `output.suffix` (lines 10-11) and defaults to `_redacted`.
* Example: `Bank1.pdf` ➔ `Bank1_redacted.pdf` when using the default config, or any custom suffix defined in profile overrides.

### Highlight output

* `n2n/renderers/pdf_highlight.py` uses the constant `HIGHLIGHT_SUFFIX = "_highlighted"` (lines 11-54) to generate artifacts.
* Highlight naming is not configurable today; the output always follows `{stem}_highlighted.pdf`.

### Intermediate artifacts

Uploaded files are written into a temporary directory under the FastAPI worker process (`n2n/api_server.py` lines 43-47). These files are deleted after each request, so the only durable artifacts that callers need to track are the redacted or highlighted PDFs described above.
