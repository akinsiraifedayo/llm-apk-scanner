# LLM-APK Scanner

Static-analysis scanner for LLM-integrated Android applications. It extracts
inference-provider credentials, system prompts, tool-call schemas and
retrieval (RAG) artefacts from APKs, and includes the AndroZoo corpus pipeline
used to measure how often they are exposed.

This is the research instrument for the MSc dissertation *Empirical Security
Analysis of LLM-Integrated Android Applications: Secret Leakage,
Prompt-Injection Surfaces, and a Mobile Threat Model for Agentic AI*
(Ifedayo Olympicson Akinsira-Olumide, MSc Computer Science, University of East
London, 2026). Version 0.1.0 is the version used for the corpus run.

## Results

A stratified sample of 1,500 AI-candidate apps was drawn from the AndroZoo
index snapshot of 13 August 2026. 1,495 were scanned (5 downloads failed) and
111 were confirmed LLM-integrated. Every figure below is in the
[anonymised findings](#anonymised-findings).

| Exposed in the 111 LLM-integrated apps | Apps | Rate | 95% Wilson CI |
|---|---:|---:|---|
| Any security-relevant artefact | 50 | 45.0% | 36.1–54.3% |
| Inference-provider credential | 39 | 35.1% | 26.9–44.4% |
| System prompt | 16 | 14.4% | 9.1–22.1% |
| RAG artefact | 2 | 1.8% | 0.5–6.3% |
| Tool schema | 1 | 0.9% | 0.2–4.9% |

Of the 24 provider-attributed credentials, 6 were live, 14 were revoked and 4
could not be verified when checked on 15 August 2026. 83.8% of the
LLM-integrated apps had at least one finding outside `classes*.dex`.

## How it works

`scan_apk()` in `llm_apk_scanner/scan.py` runs three stages on each APK:

1. **Extraction** (`apk_loader.py`, `native_strings.py`). Androguard decodes
   the manifest. The loader then reads the string table of every
   `classes*.dex`, printable strings from bundled native libraries, and every
   asset and resource that contains recoverable text. That includes React
   Native/Hermes bundles and `resources.arsc`. Bytecode is never decompiled or
   executed.
2. **Population filter** (`filters/`). An app counts as LLM-integrated if it
   references an inference endpoint such as `api.openai.com`. An LLM SDK
   namespace also counts, but only when an invocation signature or
   request-body shape corroborates it.
3. **Detectors** (`scanners/`). These run on every app, whether or not it
   passed the filter.

| Detector | Finds |
|---|---|
| `secrets.py` | Provider credentials (OpenAI, Anthropic, Google, AWS, Hugging Face, Replicate, Groq, xAI, DeepSeek, Cohere, Perplexity, OpenRouter, Fireworks), Google service-account addresses, bearer tokens, and Slack and GitHub tokens. Confidence is reduced for placeholders and low-entropy strings. |
| `prompts.py` | System prompts and user-selectable prompt presets, in English and 14 other languages including Chinese, Japanese, Korean and Arabic. |
| `schemas.py` | Tool and function-calling declarations. |
| `rag.py` | Vector stores, embedding corpora and retrieval SDK references. |

A Google `AIza…` key is attributed to Gemini only when the same APK
references a Gemini or Vertex endpoint. Otherwise it is reported as a generic,
medium-severity secret. Repeated occurrences of the same artefact are
collapsed into one finding. Reports truncate evidence to 60 characters, so
shorter credentials still appear in full. Treat every report as sensitive.

```
llm_apk_scanner/
├── apk_loader.py      APK -> (source file, text) pairs
├── native_strings.py  strings(1)-style extraction from native libraries
├── scan.py            filter + detectors -> ScanResult
├── models.py          result schema
├── stats.py           corpus aggregation and Wilson intervals
├── cli.py             command-line interface
├── filters/           lexical and structural LLM-integration filter
├── scanners/          secrets, prompts, schemas and RAG detectors
├── reporters/         JSON, Markdown and HTML reports
├── corpus/            AndroZoo client, sampling frame, stratified sample, ledger, streaming runner
├── validation/        blind validation sample, labelling evidence, accuracy metrics
└── validators/        opt-in credential liveness checks
scripts/               analysis scripts behind the dissertation's results
tests/                 pytest suite with seeded synthetic fixtures
findings/              anonymised results of the study
```

## Installation

You need Python 3.10 or later. `scripts/label_validation_set.py` needs 3.11 or
later. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                                  # 253 tests
pytest -k test_detects_all_positive     # seeded recall floor (>= 85%)
```

`scripts/build_figures.py` also needs `matplotlib`.

## Usage

`llm-apk-scanner` and `python -m llm_apk_scanner` are equivalent. Every command
accepts `--help`.

### Scanning

```bash
llm-apk-scanner scan app.apk                             # Markdown report on stdout
llm-apk-scanner scan app.apk --format json -o app.json   # json | md | html
llm-apk-scanner filter apks/                             # list the LLM-integrated APKs in a directory
llm-apk-scanner pipeline apks/ -o results/run1           # JSON and Markdown per LLM-integrated app, plus _aggregate.json
llm-apk-scanner aggregate results/run1 -o results/stats  # prevalence with Wilson intervals (JSON and CSV)
```

### Building a corpus from AndroZoo

Downloading requires an [AndroZoo](https://androzoo.uni.lu/) API key, which
is read from the `ANDROZOO_API_KEY` environment variable. Keep it in `.env`,
which is git-ignored.

| Command | Purpose |
|---|---|
| `corpus frame [--index FILE] [-o FILE] [--per-apk]` | Filter the AndroZoo index to AI-candidate apps, one row per package unless `--per-apk` is given. Streams `latest.csv.gz` when `--index` is omitted. |
| `corpus sample --metadata FILE [--total N] [--seed N] [--years RANGE] [-o FILE]` | Draw a stratified sample (year × store) from an uncompressed, AndroZoo-format metadata CSV. |
| `corpus run [--manifest FILE] [--ledger FILE] ...` | Download, scan and delete each APK. The run can be resumed from the SQLite ledger. |
| `corpus status [--ledger FILE]` | Progress, LLM-integration rate and provider counts. |
| `corpus retry [--ledger FILE] [--rescan]` | Put failed APKs back in the queue. `--rescan` also resets APKs that were already scanned. |
| `corpus download [--manifest FILE] [-o DIR]` | Download a whole manifest to disk, without scanning. |

`corpus run` stages at most `--max-staged` APKs at a time (default 16) and
deletes each one after scanning. Peak disk use is therefore
`max_staged × APK size`, however large the corpus is.
`--keep-with-findings`, `--keep-negatives N` and `--keep-budget-gb` keep a
bounded subset of APKs for re-scanning and manual audit. Every scan is
appended to a findings log (`--findings`). Apps that are LLM-integrated or
have findings also get a gzipped detail record in `--detail-dir`.

### Validation

```bash
# Blind worksheet and separate answer key
llm-apk-scanner validate sample --findings results/findings.jsonl -o validation

# Label the sample without seeing the scanner's predictions
python scripts/label_validation_set.py --apks <apk-dir> --worksheet validation/sample_labels.csv

# Precision, recall and F1 with Wilson intervals
llm-apk-scanner validate score --labels validation/manual_labels.jsonl --key validation/sample_key.csv

# Credential liveness. Dry run by default; --live makes the calls.
llm-apk-scanner validate credentials --staging creds.csv [--live]
```

`validate credentials` reads a CSV with the columns
`provider,key,sha256,package`. With `--live`, each distinct key is sent once,
and only to a non-revocable listing or identity endpoint (OpenAI, Anthropic,
Google Gemini, Hugging Face). Keys for other providers are not checked.

## Reproducing the study

| Frozen input | Value |
|---|---|
| AndroZoo index snapshot | 13 August 2026 (3,509,911,104 bytes, 27,589,444 rows) |
| Index SHA-256 | `be75ca115cbc776d2c2f7c09345257fafac42427637474cc9773483b0738c004` |
| Sampling seed | `20260813` |
| Scanner version | `0.1.0` |

The study's own sampling frame (16,364 AI-candidate apps) and manifest (the
1,500 drawn from them) are not included here. Both accompany the dissertation
submission.

Step 1 rebuilds a frame from an AndroZoo index snapshot; with the snapshot
digest above, it reproduces the study's frame exactly. Step 2 redraws the
sample from it. Both write to `results/`, which is git-ignored.

```bash
# 1. Build the frame from the archived index snapshot
llm-apk-scanner corpus frame --index corpus/androzoo_index_20260813.csv.gz -o results/frame_20260813.csv

# 2. Draw the sample from the frame
python -c "
from pathlib import Path
from llm_apk_scanner.corpus.frame import read_frame
from llm_apk_scanner.corpus.manifest import write_manifest
from llm_apk_scanner.corpus.sample import stratified_sample
frame = read_frame(Path('results/frame_20260813.csv'))
write_manifest(stratified_sample(frame, total=1500, seed=20260813), Path('results/manifest_20260813.csv'))
"

# 3. Corpus run (needs ANDROZOO_API_KEY)
llm-apk-scanner corpus run \
  --manifest results/manifest_20260813.csv \
  --ledger corpus/ledger_v2.db \
  --findings results/findings_v2.jsonl \
  --detail-dir results/detail_v2 \
  --download-timeout 900 \
  --keep-with-findings --keep-negatives 60 --keep-budget-gb 45
```

Run the analysis scripts from the repository root after the corpus run. They
read the ledger (`corpus/ledger_v2.db`), the per-app records
(`results/detail_v2/`) and the APKs retained in `corpus/staging/`.

| Script | Output |
|---|---|
| `adversarial_audit.py APK...` | A second extraction with a wider vocabulary than the detectors, reporting every disagreement with the scanner. Prints raw matched strings to stdout; the study redirected them to `audit_full.txt`. |
| `classify_prompts.py` | `validation/prompt_classes.json`: sorts each prompt finding into system prompt or user preset. |
| `run_liveness_validation.py` | `validation/liveness_results.json`: re-extracts the provider credentials and checks whether they are live. **Makes live calls unless `--dry-run` is given.** |
| `build_chapter6_stats.py` | `validation/chapter6_stats.json`: every Chapter 6 statistic except the pilot validation. Also reads `results/findings_v2.jsonl`, the frame, `audit_full.txt` and the two files above. Stops with an error if the findings log, per-app records and ledger disagree. |
| `build_figures.py` | The dissertation figures (PDF and PNG), written to `dissertation/figures/` from `validation/chapter6_stats.json`. |
| `framework_census.py` | `validation/framework_census.json`: the build toolchain of each LLM-integrated app. The ledger and APK paths are hard-coded to the original machine. |
| `anonymise_findings.py` | `findings/`: copies of the four `validation/` files above with every application identifier removed. |

## Anonymised findings

`findings/` holds the study's results with nothing that identifies an
application.

| File | Contents |
|---|---|
| `chapter6_stats.json` | Every Chapter 6 statistic except the pilot validation of the integration filter, which is not released. |
| `liveness_results.json` | Provider, fingerprint and disposition of each of the 24 checked credentials. A fingerprint is the first 16 hex digits of the key's SHA-256. |
| `prompt_classes.json` | Classification of the 24 apps the prompt detector flagged, for each app and each string. |
| `framework_census.json` | Build-toolchain counts for the 111 LLM-integrated apps. |

`anonymise_findings.py` makes these files from the identified versions:
- **Removed:** package names, SHA-256s, key values and prompt text.
- **Reduced:** source file paths become container types (DEX, native library,
  asset, resource table), and check times become dates.
- **Reordered:** records are sorted by their own content, never by app, so
  they can't be matched against the manifest.

There is no per-app findings table. Combined with the manifest's year and
store columns, the smaller strata could narrow a finding down to a few named
apps.

The dissertation's figures can be regenerated from these files:

```bash
mkdir -p validation && cp findings/chapter6_stats.json validation/
python scripts/build_figures.py      # writes dissertation/figures/
```

## Not included

No file in this repository names a sampled application or ties any
application to a finding. The only package names present are test inputs for
the name filter.

- **The sampling frame and manifest.** They accompany the dissertation
  submission.
- **APKs.** AndroZoo's terms prohibit redistributing them.
- **The AndroZoo index snapshot** (3.5 GB). Check any copy against the
  SHA-256 above.
- **Scan outputs and identified analysis files.** This covers the ledger,
  findings log, per-app records and audit transcript, plus the original
  versions of the `findings/` files. They are held privately under the
  study's responsible-disclosure protocol.

`.gitignore` excludes all of these: `results/`, `validation/`, `corpus/`,
`.env`, the index snapshot and the audit transcript. Scan outputs and app
lists therefore can't be committed by accident.

## Responsible use

- Scanning makes no network calls, and credentials found in an APK are never
  used.
- Credential checks are opt-in and limited to the introspection endpoints
  listed above.
- Report an exposed key to the app's developer or to the provider. Do not
  test it further.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@mastersthesis{akinsiraolumide2026llmapk,
  author = {Akinsira-Olumide, Ifedayo Olympicson},
  title  = {Empirical Security Analysis of {LLM}-Integrated Android Applications:
            Secret Leakage, Prompt-Injection Surfaces, and a Mobile Threat Model
            for Agentic {AI}},
  school = {University of East London},
  type   = {{MSc} dissertation},
  year   = {2026}
}
```
