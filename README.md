# N2N — a fail-closed disclosure gate

N2N takes a sensitive document plus a declared purpose, and either returns
a certified, purpose-minimised, irreversibly redacted document with
machine-readable proof of what it did — or it refuses and tells you
exactly why.

**Every document leaves with proof — or it doesn't leave.**

## Phase 1 scope

Native-text PDFs only — scanned/photographed documents and OCR are out of
scope for Phase 1. Structured identifiers only, each checksum- or
label-validated, never a bare pattern match: UK sort codes, account
numbers, GB IBANs (mod-97), payment card numbers (Luhn, clearly
formatted — standard 4-4-4-4 and Amex's 4-6-5 grouping), and card expiry
dates. Free-text name/address candidates are always review-tier — never
auto-resolved (see `n2n/detectors/name_header.py`).

Two purpose packs exist, sharing the same detectors/pipeline/transform/
verification machinery — depth on one proven engine, not a second one:

- **`uk.bank_statement.share_with_ai`** — sort code, account number,
  IBAN, card number, card expiry.
- **`pci.card_data.share_with_ai`** — card number and expiry, for any
  document carrying payment-card data (receipts, order confirmations,
  cardholder forms), not just bank statements. Does not detect CVV/CVC —
  those should never be present on a stored document under PCI DSS, and a
  reliable label-free detector for a bare 3-4 digit code doesn't exist
  yet.

Purpose packs beyond these two, and anything requiring a vision/ML model
(e.g. face or ID-card detection), are out of scope for now (see the build
spec, sections 8 and 10) — the latter also carries a licensing trap the
spec explicitly flags: avoid Ultralytics YOLO (AGPL-3.0) if that's ever
added.

## The five release states

`PASS_AUTO`, `NEEDS_REVIEW`, `UNSUPPORTED`, `FAILED_VERIFY`,
`PROCESSING_ERROR`. No other return state exists. It is structurally
impossible for the CLI to emit an output file on any status other than
`PASS_AUTO` — see `n2n/output_gate.py` for how that invariant is enforced
(and `tests/test_output_gate.py` for how it's guarded against regressions).

## Pipeline

1. **Preflight** (`n2n/preflight.py`) — classify the input; reject anything
   outside Phase 1's supported class immediately. If classification fails,
   one genuine repair attempt via pikepdf/QPDF (`n2n/repair.py`) — far
   more capable than anything hand-rolled here at recovering a
   structurally damaged PDF — before giving up; if it helps, the pipeline
   proceeds on the repaired bytes, and that's recorded transparently in
   the manifest's `extraction_methods` (never silent).
2. **Font trust check** (`n2n/font_trust.py`) — a font with no ToUnicode
   CMap, or an embedded font program that doesn't even parse, means
   extracted text can't be trusted to match what's rendered (the literal
   bug class behind the Epstein-files/Meta v. FTC redaction failures the
   spec cites). Routes straight to `NEEDS_REVIEW` before extraction runs,
   using pikepdf for PDF-structure inspection and fontTools to validate
   the embedded font program.
3. **Extraction** (`n2n/extract.py`) — native text, layout, metadata, forms,
   annotations, incremental-update history, via PyMuPDF.
4. **Detection** (`n2n/detectors/`) — structured detectors first
   (checksum/label-validated), then free-text name candidates, tagged
   separately by confidence tier.
5. **Policy resolution** (`n2n/policy.py`) — apply the pack's must-hide /
   must-preserve rules; any conflict forces `NEEDS_REVIEW`.
6. **Transform** (`n2n/transform.py`) — irreversible removal via content
   stream rewrite (PyMuPDF redaction annotations, not an overlay box),
   plus metadata/form/annotation/embedded-file stripping and history
   flattening.
7. **Independent verification** (`n2n/verify.py`) — reopens the *output*
   through pdfplumber, a separate library from the one used to extract
   and transform, and checks for residual matches.
8. **Evidence manifest** (`n2n/manifest.py`) — signed with a local Ed25519
   keypair (`n2n/keys.py`, generated on first run, customer-owned — no
   hosted signing service).
9. **Release decision** — only `PASS_AUTO` releases a file.

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

## Web UI (`n2n serve`)

```bash
n2n serve --port 8000
```

Runs a local upload -> instant result -> findings -> download flow at
`http://127.0.0.1:8000`, entirely in-process: no document content or
telemetry leaves the machine, and nothing is persisted beyond an
in-memory, TTL-expiring session that exists only so a `PASS_AUTO` result
can be downloaded (`n2n/webapp/sessions.py`).

- **`GET /v1/packs`** — list purpose packs.
- **`POST /v1/redact?pack_id=...`** (multipart `file`) — runs the full
  pipeline and returns the decision report as JSON, plus a
  `download_token` when (and only when) the status is `PASS_AUTO`.
- **`GET /v1/download/{token}/output.pdf`** / **`/manifest.json`** —
  download the certified output and its signed manifest. 404 for any
  token that doesn't correspond to a live `PASS_AUTO` session — there is
  no code path that can serve a file for a refused document.

The findings the API returns deliberately never include the underlying
sensitive text — only field type, page, geometry, confidence tier, and
what happened to it (`n2n/models.py:DecisionReport.to_dict`,
covered by `tests/test_webapp.py::test_findings_never_include_raw_sensitive_text`).
The UI shows per-finding outcomes distinctly for a refusal
(*"Would remove"* / *"Flagged for review"*) versus a certification
(*"Removed"*) rather than one blended state, and never renders a
confidence score anywhere — both required by spec section 6.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers detector validators (mod-97, Luhn), the output-gate invariant
(including a static check that only `n2n/pipeline.py` may mint a release
token), the web API (`tests/test_webapp.py` — download gating, 404s for
unknown/forged tokens, no raw sensitive text in responses), full pipeline
runs for `PASS_AUTO`, `NEEDS_REVIEW`, and
`UNSUPPORTED`, signed-manifest verification, and deterministic replay
(same input + pack + engine version → byte-identical output).

## Adversarial testing (`tests/test_adversarial.py`)

A first pass at trying to break the engine with hostile-but-still-native-text
PDFs, not the full Phase 2 corpus (spec section 8/9 — a few hundred documents,
real bank layouts, published per-field benchmarks). It found and fixed four
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
4. **Dash-variant separators.** A sort code printed with a font that
   substitutes an en dash, em dash, minus sign, or other dash-like glyph
   for a plain hyphen ("12–34–56") matched nothing, since the value regex
   only accepted ASCII `-` — another silent leak. Fixed by widening the
   separator match to a small set of dash-like characters, applied
   consistently in both the detector's scan pattern and its normalizer.

**Previously a known gap, now closed:** a font with a broken or missing
`ToUnicode` CMap, where the extracted text doesn't match the rendered
glyphs at all (the literal bug class behind the Epstein-files and Meta v.
FTC redaction failures cited in the build spec). pikepdf (direct
PDF-structure editing) and a synthetic minimal font built in-process with
fontTools (no external file dependency) made an actual repro possible —
see `n2n/font_trust.py` and `tests/test_font_trust.py`. A document with
such a font now routes to `NEEDS_REVIEW` before detection even runs,
rather than silently trusting text that might not match what's on the
page. This doesn't prove a *present* ToUnicode mapping is semantically
correct, only that the two cheap, well-defined failure modes it checks
(missing entirely, or an unparseable embedded font program) aren't
present — a deliberately-wrong-but-present mapping is a harder, deeper
version of this same bug class and remains unaddressed.

## Determinism testing (`tests/test_transform_id_determinism.py`)

The deterministic-replay guarantee (spec 5.7) was itself found broken by
stress-testing rather than by inspection: a two-run comparison
(`tests/test_pipeline.py`'s original replay test) passed reliably, but
running the *same* input through the pipeline 200-400 times in a loop
produced up to 3 distinct output files. Root cause: MuPDF stamps a fresh,
random trailer `/ID` on every save (the second of its two entries is
*meant* to change per revision, per PDF spec convention), and can encode
either entry as a hex string (`<...>`) or a PDF literal string (`(...)`,
with backslash escapes) — the original neutralization in
`n2n/transform.py` only recognized the hex form via regex, and silently
left MuPDF's own random ID untouched whenever it chose the other one.
Fixed with a proper parser for both PDF string forms (`_find_pdf_string_end`
in `n2n/transform.py`), plus a 40-iteration stress test
(`test_deterministic_replay_holds_over_many_runs`) so a bug with this kind
of low, non-uniform trigger rate gets caught by CI odds, not by luck.

## Malformed-PDF repair and font trust (`n2n/repair.py`, `n2n/font_trust.py`)

Two hardening passes built on well-established open-source PDF tooling
rather than hand-rolled parsing:

- **Repair.** A document preflight can't classify gets one genuine second
  attempt via pikepdf/QPDF before being refused — QPDF's repair handling
  is far more battle-tested than anything built here for xref-table
  damage or dangling object references. In practice PyMuPDF's own
  repair-on-open already recovers most simple truncation, so this mostly
  helps a narrower class of structural corruption; the orchestration
  logic (try repair, re-classify the result, use it only if it's now
  genuinely supported, clean up the temp file either way) is covered
  directly in `tests/test_repair.py`. Never silent: a repaired document's
  manifest records `pikepdf_repair_applied` in `extraction_methods`.
- **Font trust.** See the "previously a known gap" note above — this is
  the fix for the `ToUnicode` gap, using pikepdf for PDF-structure
  inspection and fontTools to validate embedded TrueType/OpenType font
  programs.
