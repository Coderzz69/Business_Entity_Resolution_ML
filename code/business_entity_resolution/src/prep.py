"""Stage 1: cache raw TSVs as parquet and run pass-1 normalisation on every source.

Training-set size limit: --max-rows, or EDA_MAX_ROWS from the project .env, caps the number of
*training* S1 entities. The S2/S3 pools keep every record matched to a sampled S1, plus the same
fraction of all other records, so the distractors-per-entity ratio (and therefore the dev score) stays
realistic. The test split is never sampled: the submission must contain every test S1 entity.
"""
import argparse, os, time
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import cache_source, load_env, load_gt
from normalize import normalize_frame


def sample_train(data_dir: Path, work: Path, max_rows: int, seed: int = 0) -> dict:
    """Returns {src: raw frame} for a sampled training split."""
    rng = np.random.default_rng(seed)
    s1 = pd.read_parquet(cache_source(data_dir, work, "train", 1))
    if max_rows >= len(s1):
        return {1: s1} | {k: pd.read_parquet(cache_source(data_dir, work, "train", k)) for k in (2, 3)}
    s1 = s1.sample(max_rows, random_state=seed)
    frac = max_rows / 2_206_821
    pairs, _ = load_gt(data_dir)
    linked = pairs.m[pairs.s1.isin(s1.id)].to_numpy()
    out = {1: s1}
    for k in (2, 3):
        d = pd.read_parquet(cache_source(data_dir, work, "train", k))
        keep = d.id.isin(linked).to_numpy() | (rng.random(len(d)) < frac)
        out[k] = d[keep]
    print(f"sampled train: S1 {len(s1):,}, S2 {len(out[2]):,}, S3 {len(out[3]):,} (other-record fraction {frac:.3f})",
          flush=True)
    return out


def main(data_dir, work, max_rows):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    raw_cache = Path(os.environ.get("RAW_CACHE", work))     # full-size parquet cache can be shared across work dirs
    train_raw = None
    for split in ("train", "test"):
        for src in (1, 2, 3):
            out = work / f"n1_{split}_s{src}.parquet"
            if out.exists():
                continue
            t = time.time()
            if split == "train" and max_rows:
                if train_raw is None:
                    train_raw = sample_train(data_dir, raw_cache, max_rows)
                raw = train_raw[src]
            else:
                raw = pd.read_parquet(cache_source(data_dir, raw_cache, split, src))
            normalize_frame(raw).to_parquet(out)
            print(f"normalised {split} S{src}: {len(raw):,} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap on training S1 entities (default: EDA_MAX_ROWS from .env; 0 = no cap)")
    a = ap.parse_args()
    mr = a.max_rows if a.max_rows is not None else int(load_env(a.env).get("EDA_MAX_ROWS", 0) or 0)
    print(f"training S1 cap: {mr if mr else 'none (full data)'}", flush=True)
    main(Path(a.data), Path(a.work), mr or None)
