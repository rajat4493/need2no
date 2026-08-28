# N2N — a fail-closed disclosure gate

N2N takes a sensitive document plus a declared purpose, and either returns
a certified, purpose-minimised, irreversibly redacted document with
machine-readable proof of what it did — or it refuses and tells you
exactly why.

**Every document leaves with proof — or it doesn't leave.**

## Phase 1 scope

Native-text UK bank statement PDFs only. Structured identifiers only: UK
sort codes, account numbers (label-validated), GB IBANs (mod-97 checksum),
payment card numbers (Luhn checksum, clearly formatted). Free-text
name/address candidates are always review-tier — never auto-resolved (see
`n2n/detectors/name_header.py`).

Scanned/photographed documents, OCR, and purpose packs beyond
`uk.bank_statement.share_with_ai` are out of scope for Phase 1 (see the
build spec, sections 8 and 10).

## The five release states

`PASS_AUTO`, `NEEDS_REVIEW`, `UNSUPPORTED`, `FAILED_VERIFY`,
`PROCESSING_ERROR`. No other return state exists. It is structurally
impossible for the CLI to emit an output file on any status other than
`PASS_AUTO` — see `n2n/output_gate.py` for how that invariant is enforced
(and `tests/test_output_gate.py` for how it's guarded against regressions).

## Pipeline

1. **Preflight** (`n2n/preflight.py`) — classify the input; reject anything
   outside Phase 1's supported class immediately.
2. **Extraction** (`n2n/extract.py`) — native text, layout, metadata, forms,
   annotations, incremental-update history, via PyMuPDF.
3. **Detection** (`n2n/detectors/`) — structured detectors first
   (checksum/label-validated), then free-text name candidates, tagged
   separately by confidence tier.
4. **Policy resolution** (`n2n/policy.py`) — apply the pack's must-hide /
   must-preserve rules; any conflict forces `NEEDS_REVIEW`.
5. **Transform** (`n2n/transform.py`) — irreversible removal via content
   stream rewrite (PyMuPDF redaction annotations, not an overlay box),
   plus metadata/form/annotation/embedded-file stripping and history
   flattening.
6. **Independent verification** (`n2n/verify.py`) — reopens the *output*
   through pdfplumber, a separate library from the one used to extract
   and transform, and checks for residual matches.
7. **Evidence manifest** (`n2n/manifest.py`) — signed with a local Ed25519
   keypair (`n2n/keys.py`, generated on first run, customer-owned — no
   hosted signing service).
8. **Release decision** — only `PASS_AUTO` releases a file.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## CLI usage

```bash
n2n packs

n2n redact statement.pdf \
  --pack uk.bank_statement.share_with_ai \
  --output safe.pdf \
  --manifest safe.n2n.json

n2n redact statement.pdf --pack uk.bank_statement.share_with_ai --dry-run
```

A non-`PASS_AUTO` run exits with a non-zero status code and writes no
output file — check `--dry-run` output or the JSON report on stdout for
plain-language reasons.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers detector validators (mod-97, Luhn), the output-gate invariant
(including a static check that only `n2n/pipeline.py` may mint a release
token), full pipeline runs for `PASS_AUTO`, `NEEDS_REVIEW`, and
`UNSUPPORTED`, signed-manifest verification, and deterministic replay
(same input + pack + engine version → byte-identical output).
