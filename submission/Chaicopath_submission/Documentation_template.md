# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Chaicopath
**Team Members:** Shreeya KS (Team Leader), Sanath Coonani, Pranav Prerepa, Sudhir
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A CPU/GPU-light pipeline:
1. **Data-driven normalisation.** Any script is transliterated to Latin letters. Synonym tables (transliterations, abbreviations, state codes) are learned automatically from matched training pairs.
2. **Conjunction-key sparse blocking.** 97.8% of true pairs survive in a top-15-per-source candidate set.
3. **Two-stage XGBoost matcher.** It uses structural signals from the problem: every S2/S3 record belongs to at most one S1, and a true S2 match should agree with the true S3 match.
4. **Decoder for the actual metric.** Each S1's match list is chosen to maximise expected per-entity F0.5.

The submitted files come from the model trained on the full train set, with held-out macro F0.5 **0.9714** (ROC AUC 0.9996, Somers' D 0.9991). No external data, APIs or pretrained models are used.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA over all 2.2M train and 1.7M test S1 records and 20M S2/S3 records:
- **Link structure is a partition.** 0 of 7.6M matched S2/S3 IDs belong to more than one S1.
- **Match counts.** 5.6% of S1 entities have no match. The mean is 3.46 matches (1.67 in S2, 1.79 in S3), and the maximum is 11.
- **Distractors.** About 26% of S2/S3 records match no S1 at all.
- **Scripts.** 23.5% of India names in S2 and 13% in S3 are written in Indian scripts (Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati). S1 names are always Latin.
- **Name noise.**
  - Junk prefixes: 13% of S2 names.
  - Web handles and domains: `@doelectronics`, `maabigconsultancy.com`.
  - Fake aliases: `Drexquoyuma a/k/a Empire Coalition Inc`.
  - Word shuffles, repeated words, character typos, dropped or added legal suffixes, and appended phone numbers.
- **Address noise.**
  - Mangled house numbers: `005510`, `43945-C`, `1-95`, `195→19`.
  - Reordered components, and states as codes, full names or Indian-script names (`MH` / `Maharashtra` / `महाराष्ट्र`).
  - A county used in place of the city, and about 3% of S2/S3 addresses empty.
- **France appears only in test** (about 15% of every test file), concentrated in about 10 cities with generic names such as "Maison Ligue SARL". Its addresses are mixed at region and department level.
- **No ID leakage.** The Spearman correlation between S1 and matched ID numbers is 0.006.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier + metric-aware structured decoding.
**Core Innovation:** Four ideas, each driven by what the data showed:
- learned synonym and transliteration tables built by aligning the residual tokens of matched pairs;
- conjunction-key blocking;
- partition-aware "reverse competition" and cross-source agreement features;
- per-entity expected-F0.5 decoding.

---

## 3. Candidate Generation (Blocking)

- **Normalisation before blocking:**
  - `anyascii` transliteration and lowercasing;
  - splitting alias markers and unwrapping domains;
  - extracting house, plot and postal numbers with leading zeros stripped;
  - the learned synonym tables. For each training pair, the residual tokens found on only one side are aligned by position, or as 1↔2 phrases, and merged with union-find when they substitute for each other in both directions. From labels alone this recovers `praivet→private`, `kmstrksms→constructions`, `mharastr/mh→maharashtra`, `north carolina→nc` and `st→street`.
  - glued handles are split with a word-frequency model built from the corpus: `maabigconsultancy → maa big consultancy`.
- **Blocking keys used.** Each record becomes a bag of hashed keys.
  - *Single keys:* core-name tokens, 4-letter prefixes of long name tokens (typo tolerance), address words, numbers, and the glued core name.
  - *Conjunction keys:* name × name, name × house number, name × the record's two rarest address words, and number × rarest address word.
  - Keys are weighted by their IDF in the S2/S3 pool. Keys with document frequency above 5,000 are dropped from the index only.
  - Top-15 S2 and top-15 S3 records per S1 are found by cosine with sparse top-n matrix products (`sparse_dot_topn`), per country.
- **Candidate pairs generated:** 51,975,952 on test, 30 per S1 entity: France 7.78M, India 24.30M, US 19.89M.
- **How we made sure true matches were not lost:** conjunction keys let us prune common tokens hard without losing records whose words are all common. Recall measured on 40k held-out India S1 records against the full 4.1M pool:

| Variant | Recall at top 5 | at top 10 | at top 15 |
|---|---|---|---|
| Single keys, pruning at 5,000 | 86.6% | 90.2% | 91.8% |
| Single keys, pruning at 2% of pool (4× slower) | 91.9% | 94.6% | 95.5% |
| **Single + conjunction keys, pruning at 5,000** | **94.2%** | **96.5%** | **97.1%** |

  Pair recall on the full-data dev set is **97.78%**.

---

## 4. Matching Model

**Features used (44 pairwise in stage A, plus 13 in stage B):**
- **Name features:** Levenshtein ratio, token-set, token-sort, partial ratio, Jaro-Winkler and common-prefix similarity on the core and glued name; token-set on the full name including legal words; alias-to-name similarity; legal-suffix agreement; token counts.
- **Address features:** token-set, token-sort, partial and ratio on the canonical address; number-set overlap; whether the first numbers are equal, and their ratio; number counts; empty-address flags.
- **Blocking and IDF features:** the blocking cosine plus separate name, address, number and glued-name IDF cosines; the candidate's rank and gap to the best candidate in its list.
- **Reverse competition (partition-aware):** how many S1 lists contain the candidate, its best score from any S1, whether this S1 is its best claimant, and the margin to the runner-up.
- **Stage B (list-level):** the stage-A probability; its rank, gap to the best and sum across the S1's list; the number of candidates above 0.5; the best score per source; and **cross-source agreement**, meaning name, address and number similarity to the best candidate from the *other* source, weighted by that candidate's probability.
- **Other:** flags for non-Latin script and web handles, and the source (S2/S3). **No country feature is used**, so the model transfers to France.

**Model type:** XGBoost (hist, depth 8, 400 rounds; CUDA on an RTX 3050, with automatic CPU fallback), in two stages. Stage B trains on stage A's out-of-fold predictions (3-fold GroupKFold grouped by S1). Isotonic calibration is fitted on stage B's out-of-fold predictions.

**Threshold selection method:** no global threshold. Instead:
1. **Partition constraint:** each S2/S3 record is kept only for its highest-probability S1.
2. **Expected-F0.5 decoding:** for each S1, candidates are sorted by probability and the list length k ≥ 0 that maximises Monte-Carlo expected F0.5 is chosen. The estimate includes a Poisson term for true matches that blocking missed, with its rate tuned on out-of-fold training predictions. This optimises the macro, per-entity F0.5 directly, including the singleton rule.

---

## 5. Results & Error Analysis

Dev = S1 entities held out by an ID hash, never used for fitting, blocked against the S2/S3 pool.

| Step | **Submitted model** (full train set, 8% fit / 4% dev) | Quick-run variant (200k S1, 70/30) |
|---|---|---|
| Blocking pair recall | 97.78% | 98.87% |
| Stage A, best fixed threshold | 0.9681 | 0.9816 |
| Stage B, calibrated, threshold 0.5 | 0.9694 | 0.9809 |
| + partition constraint | 0.9694 | 0.9809 |
| **+ expected-F0.5 decoding (final)** | **0.9714** | **0.9825** |
| ROC AUC / Somers' D (final probability) | 0.9996 / 0.9991 | 0.9998 / 0.9997 |

- **F0.5 Score (macro):** **0.9714** for the submitted model on full-data dev. The quick-run variant's 0.9825 is optimistic because its sampled pool keeps only 9.1% of other entities' records, so it contains fewer look-alikes. It is shown only for comparison.
- **Submitted test output:** 5,764,109 matches (3.33 per S1); 5.5% of S1 entities predicted as singletons (train: 5.6%). Passes the official validator with `--check-ids`.
- **Overfitting check (full-data model):**
  - In-sample F0.5 is 0.9775, out-of-fold on training data 0.9708, and dev 0.9714. The last two agree.
  - Dev log-loss is still falling at round 400 for both stages.
  - Dev vs test: mean probability 0.113 vs 0.117, and matches per S1 3.26 vs 3.33.
- **Common false negatives (missed matches):**
  - S2/S3 records with an empty address and a truncated or rewritten name (`frontier` vs `frontier regional`, `capital` vs `capital initiative`);
  - heavy double typos (`anilan inn ivg tins`);
  - city renamings (`calcutta`/`kolkata`).

  These are mostly the 2.2% that blocking never proposes.
- **Common false positives (wrong merges):** not yet broken down by category. The main risks by design are records that share a street or building with a similar generic name, including the generic French names concentrated in a few cities. The partition constraint and competition features target this class. A categorised false-positive audit is the next step.

---

## 6. Conclusion

- Learning normalisation from the labels, instead of hand-writing it, handled the Indian-script and abbreviation noise.
- Conjunction-key blocking gave near-embedding recall at a fraction of the cost.
- Encoding the problem's structure (the partition rule, cross-source agreement, and the per-entity F0.5 objective) gave the final gains without any large model.
- Lessons: measure blocking recall first; decode for the real metric; watch memory. RAM, not compute, was the binding constraint on a 15 GB machine.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: run `bash run_pipeline.sh <dataset> <work> <output>`. The defaults (full train set, `TRAIN_PCT=8`, `DEV_PCT=4`) reproduce the submitted files; the README explains the settings.

| Script | Stage |
|---|---|
| `src/prep.py` | Parquet cache and pass-1 normalisation (`normalize.py`); optional training-size cap |
| `src/canon.py` | Learns synonym tables (`synonyms.py`) and canonicalises all sources |
| `src/block.py` | Candidate generation (`blocking.py`) and competition features |
| `src/train.py` | Two-stage XGBoost, calibration, decoder tuning and dev report (`features.py`, `decode.py`) |
| `src/predict.py` | Test scoring and decoding; writes `matching_results.tsv` and `candidate_pairs.tsv` |
| `src/validate_submission.py` | Official validator (run as the final step) |
| `src/evaluate.py`, `src/overfit_check.py` | ROC / Somers' D, train vs dev, learning curves, drift |

### B. Additional Results

- `code/business_entity_resolution/docs/roc_full.png`: ROC curves on dev for the blocking score, stage A and stage B.
- `code/business_entity_resolution/docs/overfit_full.png`: learning curves for both stages, and the probability distribution on dev vs test (all and France).
- Per-slice final-model AUC on full-data dev: India 0.9995, US 0.9996, S2 0.9996, S3 0.9995.
