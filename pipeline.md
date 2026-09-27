# Business Entity Resolution: pipeline

The task is to find, for every Source 1 (S1) business record, all records in Source 2 (S2) and Source 3 (S3) that describe the same business. The score is F0.5, computed per S1 entity and then averaged. Precision counts twice as much as recall, and S1 entities with no matches count too: an empty prediction for them scores 1.0.

The pipeline runs on CPU only (12 cores, 15 GB RAM) and uses no external data, APIs or pretrained models. All code is in `code/business_entity_resolution/src/`.

```bash
bash code/business_entity_resolution/run_pipeline.sh dataset work output
```

---

## 1. What the data told us

| Finding | Consequence for the design |
|---|---|
| Train: 2.2M S1 against 5.0M S2 and 5.3M S3. Test: 1.73M S1 against 4.9M S2 and 5.1M S3. | Comparing all pairs means about 10¹³ comparisons, so a cheap candidate-finding step (blocking) comes first. |
| Each S2/S3 record belongs to **at most one** S1 entity (0 exceptions among 7.6M matched IDs). | This rule is enforced at the end and is also given to the model as features. |
| Only 5.6% of S1 entities have no match. The average is 3.46 matches and the maximum is 11. | Missing true matches costs points too, not only false ones. |
| About 26% of S2/S3 records match nothing. | These distractors are the main source of false matches. |
| 23.5% of India names in S2 and 13% in S3 are written in Indian scripts. | Everything is transliterated to Latin letters, and the transliterations are then mapped back to English words. |
| Names come with junk prefixes, web handles (`maabigconsultancy.com`), fake aliases (`X a/k/a Real Name`), typos, shuffled word order and repeated words. | Name cleaning handles these; similarity is measured in several ways. |
| House numbers are mangled (`005510`, `43945-C`, `1-95`). Address parts are reordered or missing, and state appears as a code or a full name. | Numbers are extracted separately, and addresses are compared as bags of tokens rather than in order. |
| France appears only in test, clustered in about 10 cities. | Nothing depends on the country label, and French street abbreviations are seeded. |

---

## 2. Pipeline stages

```
TSV ──► prep ──► canon ──► block ──► train (fits on train split) ──► predict (test) ──► 2 TSVs + validator
```

### Stage 1: `prep.py`, text cleaning pass 1 (`normalize.py`)
- Caches each TSV as parquet, with IDs converted to integers.
- Transliterates any script to ASCII (`anyascii`) and lowercases.
- Splits fake aliases (`a/k/a`, `dba`, `f/k/a`, `formerly`) from the real name.
- Unwraps web handles and domains, drops phone numbers appended to names, and extracts house, plot and postal numbers with leading zeros stripped.

### Stage 2: `canon.py`, learned synonyms (`synonyms.py`)
- For every true training pair, it takes the words that appear on only one side and lines them up:
  - lists of equal length are paired position by position;
  - a single word against two neighbouring words becomes a phrase rule.
- Substitutions that recur often, in both directions, are merged, and each group maps to its most common spelling.
- This learns a few hundred rules from labels alone:
  - transliterations: `praivet→private`, `kmstrksms→constructions`, `imjiniyrimg→engineering`
  - states: `mharastr / mh→maharashtra`, `texas→tx`, `north carolina→nc`, `uttar pradesh→up`
  - streets: `st→street`, `rd→road`
- Glued names are split into words using a word-frequency model built from the corpus: `maabigconsultancy → maa big consultancy`.
- Each name ends up as core words, legal-form words (`private`, `llc`, `sarl`, …) and a glued "compact" form.

### Stage 3: `block.py`, candidate generation (`blocking.py`)
- Each record becomes a bag of keys:
  - **Single keys:** name words, 4-letter prefixes of long name words (for typos), address words, numbers, and the compact name.
  - **Combination keys:** name word × name word, name word × house number, name word × the record's rarest address word, and number × rarest address word.
- Keys are weighted by how rare they are in the S2/S3 pool (IDF). Keys found in more than 5,000 records are left out of the search index.
- Sparse matrix multiplication (`sparse_dot_topn`) keeps the **top 15 from S2 and the top 15 from S3** for each S1. This runs per country, never across countries.
- Combination keys are what make aggressive pruning safe: a "Global … Lucknow" record stays findable even though `global` and `lucknow` are each too common to index alone.
- Context features added here:
  - the candidate's rank and score gap within its S1's list;
  - **competition**: how strongly other S1 records claim the same S2/S3 record (its best score anywhere, whether this S1 is its best claimant, the margin, and how many S1 lists contain it).

Measured on 40k held-out India S1 records, against the full 4.1M pool:

| Blocking variant | Recall at top 5 | at top 10 | at top 15 |
|---|---|---|---|
| Single keys only, pruning at 5,000 | 86.6% | 90.2% | 91.8% |
| Single keys only, pruning at ~2% of pool (slow) | 91.9% | 94.6% | 95.5% |
| **Single + combination keys, pruning at 5,000** | **94.2%** | **96.5%** | **97.1%** |

Full train set: 66.2M candidate pairs, in about 4 minutes for India and 6 minutes for the US.

### Stage 4: `train.py`, two-stage matcher (`features.py`)
- **Split:** S1 entities are assigned to buckets by a hash of their ID. 10% are used for fitting, 4% are held out as the dev set, and neither overlaps the other.
- **Stage A features:** 44 pairwise features, computed with rapidfuzz's `cpdist`, which is multithreaded C++:
  - name: ratio, token-set, token-sort, partial, Jaro-Winkler and prefix similarity on the core and compact name, alias-to-name similarity, legal-form agreement, word counts;
  - address: token-set, token-sort, partial and ratio; number-set overlap; whether the first numbers are equal;
  - per-view IDF cosines (name, address, numbers, compact) from blocking;
  - the context and competition features from Stage 3;
  - flags for empty addresses, non-Latin script and web handles.

  **No feature uses the country label**, so the model can transfer to France.
- **Stage B features:** stage-A probabilities for the whole candidate list:
  - rank, gap to the best candidate, number of candidates above 0.5, and the best score from each source;
  - **cross-source agreement**: how similar the candidate is to the best candidate from the *other* source, weighted by that candidate's probability. A true S2 match should look like the true S3 match, because both are noisy copies of the same business.
- Both stages use XGBoost (`device="cuda"` when an NVIDIA GPU is available, CPU otherwise) with 3-fold GroupKFold grouped by S1. Stage B trains on stage A's out-of-fold predictions, so it never sees probabilities from a model that trained on the same rows.
- Probabilities are calibrated with isotonic regression, fitted on stage B's out-of-fold predictions.

### Stage 5: `predict.py`, decoding for the metric (`decode.py`)
1. **One S1 per record:** each S2/S3 record is kept only for the S1 that gives it the highest probability.
2. **Expected-F0.5 decoding:** for each S1, candidates are sorted by probability, and the list length `k` (including `k = 0`, an empty list) is chosen to maximise expected F0.5. The expectation is estimated by Monte Carlo over the calibrated probabilities, plus a Poisson term for true matches that blocking missed, with its rate tuned on the training split's out-of-fold predictions. This optimises the per-entity average directly, singletons included, which one global threshold cannot do.
3. It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (all matches are a subset of the candidates) and runs `validate_submission.py`.

---

## 3. Results

Full-data run: trained on 8% of train S1 entities (7.9M candidate pairs, XGBoost on an RTX 3050 GPU). Scored on a separate 4% of S1 entities that were never used for fitting. The blocking search ran against the full S2/S3 pool, distractors included.

| Step (dev set = 4% of train S1) | Macro F0.5 |
|---|---|
| Blocking pair recall (share of true pairs among the candidates) | 97.78% |
| Stage A, threshold 0.3 | 0.9513 |
| Stage A, threshold 0.5 | 0.9649 |
| Stage A, threshold 0.7 | 0.9681 |
| Stage B (list + cross-source features), calibrated, threshold 0.5 | 0.9694 |
| + one S1 per record | 0.9694 |
| **+ expected-F0.5 decoding (final)** | **0.9714** |

What each step adds:
- **Stage B** gains +0.0013 over the best stage-A threshold, and +0.0045 at the same 0.5 threshold.
- **The one-S1-per-record rule** adds nothing measurable here. The dev set holds only 4% of S1 entities, so there are few competing S1 records within it. On the test set, where every S1 competes, it should matter more.
- **Expected-F0.5 decoding** adds +0.0020 over the best fixed threshold. It also needs no threshold tuning.
- The training split's own out-of-fold score is 0.9708, close to dev, so there is no sign of overfitting.

The end-to-end smoke test on a small sample passes every stage and the official validator. Its scores (~0.988) are inflated by the small distractor pool and should not be quoted.

---

## 4. Compared with the "Hybrid Stacked" plan

| | Hybrid plan | This pipeline |
|---|---|---|
| Blocking | Embedding nearest-neighbour search (bge/gte-small) plus token overlap | Sparse IDF search over single and combination keys, 97% recall at top 15 in minutes on CPU |
| Indian scripts | Not addressed | Learned transliteration tables |
| Borderline pairs | 7–8B LLM judge | Stage-B list and cross-source agreement features, with no GPU needed |
| Licence risk | Llama-3.1-8B is **not** MIT/Apache, which breaks rule 5 | Only MIT/BSD/Apache libraries, no pretrained model |
| Decision | One global threshold, then greedy de-duplication | One S1 per record, then per-entity expected-F0.5 decoding |

**Possible later addition:** if a GPU becomes available, Qwen2.5-7B-Instruct (Apache 2.0) could re-rank only the few thousand most uncertain pairs. Keep it only if it raises the dev score.

---

## 5. Known limitations and next steps
- **France has no labels.** Its quality can't be measured directly. A proxy is to train on one country and evaluate on the other.
- **The blocking ceiling is ~97%.** The remaining misses are mostly S2/S3 records with an empty address and a truncated or rewritten name.
- **Memory:** stage B training on 8% of train S1 fits in 15 GB of RAM (at 10% it was killed for running out of memory). Use `--train-pct` to scale it.
