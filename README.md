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

## Adversarial testing (`tests/test_adversarial.py`)

A first pass at trying to break the engine with hostile-but-still-native-text
PDFs, not the full Phase 2 corpus (spec section 8/9 — a few hundred documents,
real bank layouts, published per-field benchmarks). It found and fixed three
real bugs before this reached a clean pass:

1. **Split-token values.** A sort code or account number whose digits land
   in separate word-tokens (odd kerning, some OCR/generator output) was
   matched against space-joined text and missed entirely — a silent leak.
   Fixed by scanning the *compact* (no-separator) reconstruction of each
   line with digit-boundary lookarounds, the same technique already used
   for IBANs.
2. **Label/value collision with statement vocabulary.** The free-text
   name-candidate heuristic matched common Title-Cased labels themselves
   ("Sort Code", "Account Number"), which would have forced `NEEDS_REVIEW`
   on nearly every realistic statement. Fixed with a label/digit exclusion
   and a small banking-vocabulary stoplist — the heuristic still stays at
   review tier only, per spec 5.6, this just improves precision.
3. **Stacked/invisible text layers.** Two independent text runs occupying
   the same coordinates (e.g. a visible line with an invisible OCR
   duplicate drawn under it) were interleaved by the y-tolerance line
   grouper, corrupting label matching for *both* layers and producing zero
   findings — the exact "visual cover survives, underlying text doesn't
   get removed" failure class the product exists to catch. Fixed by
   detecting cross-block x-overlap within a line bucket and splitting by
   content-stream block when it's found.

**Known gap, not yet covered:** a font with a broken or missing
`ToUnicode` CMap, where the extracted text doesn't match the rendered
glyphs at all (the literal bug class behind the Epstein-files and Meta v.
FTC redaction failures cited in the build spec). Constructing a minimal
repro needs a deliberately malformed embedded font rather than PyMuPDF's
standard text insertion, so it's flagged here for the Phase 2 corpus
rather than solved in this pass.
