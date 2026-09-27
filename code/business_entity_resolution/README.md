# Business Entity Resolution: reproducible pipeline

Tested on 12 CPU cores, 15 GB RAM and an optional NVIDIA GPU. Python 3.14; pinned versions are in `requirements.txt`. No external data, APIs or pretrained models are used.

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
PYTHON=python bash run_pipeline.sh /path/to/dataset work output
# -> output/matching_results.tsv, output/candidate_pairs.tsv, then the official validator runs
```

`/path/to/dataset` must contain `train/` and `test/` with the original TSVs. The defaults reproduce the submitted files:

| Setting | Default | Meaning |
|---|---|---|
| `MAX_ROWS` | `0` (full train set) | Optional cap on training S1 entities for quick runs (e.g. `200000`, with `TRAIN_PCT=70 DEV_PCT=30`). S2/S3 keep every matched record plus the same fraction of all others, so the distractor ratio stays realistic. |
| `TRAIN_PCT` / `DEV_PCT` | `8` / `4` | Share of train S1 entities used to fit / held out as dev (hash of the S1 ID, deterministic). 8% is about 177k entities and 7.9M candidate pairs, the most stage B fits in 15 GB RAM. |

The test split is never sampled. On a 15 GB machine, run it as
`systemd-run --user --scope -p MemoryMax=10G bash run_pipeline.sh ...` so a memory spike can only kill the job.
Runtime is about 1.5 hours on 12 CPU cores plus an RTX 3050 (blocking about 20 minutes, training about 16, test scoring about 18). XGBoost trains on CUDA when available and falls back to CPU automatically. Peak RAM is about 9 GB.

Diagnostics (optional): `src/evaluate.py` (ROC AUC / Somers' D) and `src/overfit_check.py` (train vs dev,
learning curves, dev vs test drift).

## Stages (`src/`)

| # | Script | What it does |
|---|--------|--------------|
| 1 | `prep.py` | Caches the TSVs as parquet with int64 IDs. Pass-1 normalisation (`normalize.py`): transliterates any script to ASCII (`anyascii`), splits `a/k/a` / `dba` / `f/k/a` aliases, unwraps web handles and domains, tokenises, and extracts house/plot/postal numbers. |
| 2 | `canon.py` | **Learns synonym tables from the training pairs** (`synonyms.py`) and canonicalises every source. Glued handles are split with a unigram-LM segmenter built on the corpus vocabulary (`maabigconsultancy → maa big consultancy`). |
| 3 | `block.py` | **Blocking** (`blocking.py`): IDF cosine over unigram and conjunction keys, top-K per S1 from S2 and from S3 via sparse top-n products, per country. Adds reverse-competition context features. |
| 4 | `train.py` | Two-stage XGBoost matcher (CUDA when available): stage A is pairwise; stage B adds list and cross-source coherence features. Then isotonic calibration, decoder tuning, and a held-out report. |
| 5 | `predict.py` | Scores test candidates, applies the partition constraint and **expected-F0.5 decoding**, and writes both files. `validate_submission.py` (the official checker) then runs as a separate step. |

## Design decisions

1. **Learned canonicalisation instead of hand dictionaries.** For each true training pair, the tokens found on only one side are aligned (by position, or as a 1↔2 phrase). Substitutions that recur in both directions are merged with union-find. This recovers, from labels alone:
   - Indic transliterations: `praivet→private`, `kmstrksms→constructions`, `imjiniyrimg→engineering`
   - state codes and phrases: `mharastr/mh→maharashtra`, `north carolina→nc`
   - street abbreviations: `st→street`

2. **Conjunction-key blocking.** Name-pair, name×number, name×rare-address-word and number×rare-address-word keys are rare and precise. Very common keys can therefore be pruned hard, which keeps the sparse top-K product fast on 5M-record pools, while a record whose words are all common is still reachable.

3. **Structure-aware matching.** The training ground truth is a partition (0 of 7.6M S2/S3 IDs belong to two S1 entities). The model sees reverse-competition features: how strongly each candidate is claimed by other S1 records. It also sees cross-source coherence: whether a candidate agrees with the best candidate from the other source, since a genuine match is a noisy copy of the same business.

4. **Decoding for the actual metric.** F0.5 is macro-averaged per S1 and singletons count, so a single global threshold is the wrong objective. For each S1 the decoder picks the prefix length (including an empty list) that maximises Monte-Carlo expected F0.5 under calibrated probabilities. Its estimate includes a Poisson term for true matches that blocking missed.

5. **France (unlabelled).** No feature uses the country label, and all rules are country-agnostic. French street abbreviations (`r→rue`, `bd`, `av`, `n°`) are seeded, and the per-view cosines use IDF from the French pool itself.
