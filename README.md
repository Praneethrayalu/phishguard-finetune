# PhishGuard

**A measured phishing-email classifier, from a strong baseline to CPU LoRA and int8 inference.**

A free, local-first Python project for learning the full ML workflow: public data ingestion, leakage-aware splitting, balanced training, TF-IDF/logistic regression, DistilBERT + PEFT LoRA, dynamic int8 quantization, reproducible evaluation and a FastAPI endpoint. No paid API keys, GPU, or cloud account needed.

> Experimental project, not a production email-security product. A "safe" prediction is never a reason to open a link. Read [MODEL_CARD.md](MODEL_CARD.md) before interpreting scores.

## Actual results, not promised results

18,650 public source emails -> 17,495 cleaned unique emails. Baseline trained on 9,136 balanced examples; LoRA experiment actually trained on 1,024. Same untouched 2,625-email test set:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| TF-IDF + logistic regression | 96.30% | 98.47% | **97.37%** |
| DistilBERT + LoRA | 87.70% | 94.69% | 91.06% |
| LoRA merged + dynamic int8 Linear | 87.14% | 92.75% | 89.86% |

Quantized inference was ~2x faster in the measured two-thread CPU batch run, with lower quality. The API defaults to the stronger and faster baseline. This project does not claim LoRA beats the baseline. See machine-readable [reports](reports/) and the model card for data lineage, split sizes, confusion matrices, timing and limitations.

## Quick start: Windows, 16 GB RAM, no GPU

Python 3.10-3.12 recommended. PowerShell, from the repository folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
phishguard prepare
phishguard baseline
pytest -q
uvicorn phishguard.api:app --host 127.0.0.1 --port 8000
```

If activation is disabled, call `.\.venv\Scripts\python.exe -m pip ...` and `.\.venv\Scripts\phishguard.exe` directly, instead of changing system-wide PowerShell policy. Linux/macOS: `python3 -m venv .venv` then `source .venv/bin/activate`. Setup downloads ~52 MB of email data. Baseline does not download a transformer. Data and artifacts are gitignored. `prepare` downloads once, creates deterministic splits and prints a manifest; `baseline` trains locally and writes measured reports. Use only trusted local model files.

Open `http://127.0.0.1:8000/docs` for the local API. PowerShell example:

```powershell
$body = @{text="Please review tomorrow's engineering meeting agenda and send comments."} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body $body
```

`GET /health` reports readiness. `POST /predict` accepts `{"text":"...email body..."}` and returns label, model, risk score and a safety warning. Inputs must be 20-100,000 characters. No body logging or outbound network calls during inference. Missing model produces HTTP 503 instead of pretending to classify. No auth/TLS built in: do not expose directly to the internet.

## Real CPU fine-tuning

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[ml]"
phishguard lora --max-samples 1024 --epochs 1 --eval-samples 128
phishguard quantize
python scripts/evaluate_models.py --model distilbert_lora_fp32
python scripts/evaluate_models.py --model distilbert_lora_int8
```

First run downloads DistilBERT (~270 MB), then saves a small LoRA adapter under `artifacts/lora/`. Training is CPU-only, two threads, batch 8, 128 tokens; suitable for a 16 GB laptop, but actual time varies. The measured 1,024-example run took ~99 seconds on the build machine, not necessarily yours. For all 9,136 balanced training examples, run `phishguard lora --epochs 1` (longer). The LoRA classifier head is trained too. Larger/full training is supported but **not claimed as executed**. `quantize` merges the adapter and writes a local trusted PyTorch artifact plus validation report. Test evaluator compares the same holdout for each model and saves separate JSON reports. Run once after final model selection; never choose training settings from test scores.

The transformer is optional and deliberately not the default API backend because its measured quality was lower. To run transformer inference in Python, use `load_adapter()` and `predict()` from `phishguard.transformer`; for the quantized local artifact see `scripts/predict_quantized.py`.

## Docker

```sh
docker build -t phishguard .
docker run --rm -p 127.0.0.1:8000:8000 phishguard
```

Build downloads the pinned dataset, trains the baseline, discards raw emails, and runs as a non-root user. Dockerfile is supplied, but Docker was not available in the build environment, so the image build is not claimed as tested. Native Python training, API integration, seven tests (including local LoRA and int8 inference), real LoRA, quantization and evaluations were run. To repeat all seven tests after training, set `$env:PHISHGUARD_TEST_ML="1"` in PowerShell then run `pytest -q`. Normal CI runs six tests and skips the artifact-dependent test.

## Design

```text
Pinned CSV -> normalize/filter -> deduplicate -> stratified train/val/test
                                              |
                                balance TRAIN only
                                  /            \
                         TF-IDF + LR      DistilBERT + LoRA
                              |                 |
                       FastAPI (default)   merge -> int8 Linear
                                  \            /
                                  measured reports
```

- Only training data fits vocabulary and models. Train/validation/test hashes are disjoint.
- Duplicate full normalized emails are removed before splitting; near-duplicates are a documented remaining risk.
- Fixed seed and input checksum; no private inbox data.
- Precision, recall, F1, ROC-AUC, average precision and confusion matrices.
- Unit/integration tests cover data conflicts, reproducibility, split isolation, API validation, missing artifacts and metrics. CI runs fast tests without downloading the dataset/models.

## Data and license

Dataset: [Cyber Cop mirror on Hugging Face](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset), declared LGPL-3.0, revision and file checksum recorded in [data manifest](reports/data_manifest.json). Base model: [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), declared Apache-2.0. Code: MIT. Neither raw emails nor trained weights are redistributed in this repository. Use data under its own terms. Public-corpus provenance, age, hidden template overlap and label noise limit what the scores establish.
