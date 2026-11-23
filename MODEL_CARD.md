# Model card

## Intended use
An educational experiment comparing classical ML, parameter-efficient fine-tuning, and CPU quantization for English email-body classification. Use as a research/demo signal, not as a spam filter, automatic blocking system, or a reason to trust a link. Not a calibrated probability of real-world harm.

## Data and lineage
The [Cyber Cop Kaggle dataset mirror](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) declares LGPL-3.0, with 18,650 rows. The source card calls it a direct copy of Cyber Cop's Phishing Email Detection dataset. Upstream labeling quality and rights were not independently audited. Download is pinned to source revision `34085a032c123ca237f314a01a67909cdea35e34`. Raw CSV SHA-256: `18ef4fff1acb8986f0ab01e83ede6025420c829656af5a377560fce17e153b97`. Dataset and model weights are not redistributed in git. Code's MIT license does not relicense third-party data or the Apache-2.0 base model.

Missing labels/text, very short or over-100k-character rows are dropped. Text gets Unicode NFKC, HTML unescape, whitespace collapse and lowercase. Deduplication uses the SHA-256 of normalized full text, removes conflicting labels, and happens before splitting. This is exact-normalized deduplication, not template/fuzzy/thread deduplication. 17,495 rows remain; no conflicting hashes found. Stratified 70/15/15 split, random seed 42. Training is then undersampled to 4,568 per class, 9,136 total. Validation 2,624 and test 2,625 preserve natural frequencies. Some majority-class training candidates are unused, not moved to test.

## Training
Baseline: word uni/bigram TF-IDF (60k maximum features, min_df=2), logistic regression, 500 maximum iterations, seed 42. Vocabulary is fit only on training text. Full training split used.

Transformer: `distilbert/distilbert-base-uncased` at revision `12040accade4e8a0f71eabdb258fecc2e7e948be` in PyTorch with PEFT LoRA, q_lin/v_lin targets, rank 8, alpha 16, dropout 0.1; pre-classifier and classifier also trainable. 739,586 trainable parameters. Actual committed experiment used a balanced 1,024-row training subset, one epoch, 128 optimizer steps, max length 128, batch 8, AdamW 2e-4, seed 42, two CPU threads. Training took 98.93 seconds on the build machine. This is a small experiment, not a claim of full-corpus transformer training. The command defaults to all balanced training rows if --max-samples is omitted.

Dynamic-int8 quantization applies to merged model Linear layers. Embeddings remain float, so the artifact is 138,717,643 bytes, not a fully int8 model. PyTorch emits a deprecation warning for torch.ao.quantization; migration to torchao is future work. Tested version is recorded in requirements-lock.txt.

## Measured evaluation
Fixed threshold 0.5, same held-out 2,625 emails, including 979 phishing emails. No test-set tuning was done.

| Model | Precision | Recall | F1 | Accuracy | Batched ms/email |
|---|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.9630 | 0.9847 | 0.9737 | 0.9802 | 0.215 |
| DistilBERT + LoRA, 1,024 train rows | 0.8770 | 0.9469 | 0.9106 | 0.9307 | 36.584 |
| Merged LoRA + dynamic int8 Linear | 0.8714 | 0.9275 | 0.8986 | 0.9219 | 18.241 |

Timing is total batch inference divided by N, excludes model load, and is not p95 request latency or a laptop performance promise. Baseline timing comes from baseline.json. Transformer reports come from the same test split. Int8 was about twice as fast in this run but lost F1, and validation probability deltas were large (max 0.596 on 128 examples). The baseline is the default serving model because it performed better and costs less. This result does not show that fine-tuning beats classical ML.

## Limits, risks and next experiments
- Random split within a single public corpus can reward collection artifacts, old vocabulary, or overlapping near-duplicate templates. High internal scores are not deployment evidence.
- No chronological, organization-held-out, cross-corpus, multilingual, adversarial, sender-authentication, URL reputation or attachment evaluation.
- Transformer sees only first 128 tokens. Tail-only attacks may be missed. Corpus may contain personal information; raw examples are not published.
- Scores are uncalibrated. False positives can block real mail; false negatives can cause loss. Human review is needed.
- Retrain and evaluate on a separate recent corpus, group near-duplicate templates, choose thresholds on validation only, calibrate scores, and assess realistic error costs before deployment.
- LoRA and int8 runtime artifacts must be produced locally. joblib and torch pickle can execute code: do not load files received from strangers.

## Privacy and security
API does not log bodies or send them externally. It does not fetch URLs or execute email content. Model download occurs only during setup/training. API has no built-in authentication, TLS, rate limiting or tenant isolation; bind to localhost or place behind a trusted authenticated gateway. Input size is bounded to 100,000 characters. Dataset download and model download require internet; inference after setup does not.
